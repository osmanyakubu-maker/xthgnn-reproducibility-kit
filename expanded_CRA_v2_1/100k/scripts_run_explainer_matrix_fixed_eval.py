"""Same-checkpoint node-explainer comparison on four public datasets.
GNNExplainer, GraphMaskExplainer and PGExplainer use PyG 2.5.3 through an
adapter that injects an edge gate into every X-THGNN message-passing layer.
The GraphMask/PGExplainer adapter exposes raw input endpoint/message features;
the learned edge score and random ranking are additional comparators.
"""
from pathlib import Path
import importlib.util,sys,os,json,time,copy,random,traceback
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent.parent; KIT=ROOT/'xthgnn-kit'; EXP=ROOT/'cra_expanded_v2_2_100k'
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('core',KIT/'code/X_THGNN_Reproduction_v3.py');core=importlib.util.module_from_spec(spec);sys.modules['core']=core;spec.loader.exec_module(core)
from torch_geometric.nn import MessagePassing
from torch_geometric.explain import Explainer,ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer,GraphMaskExplainer,PGExplainer
from torch_geometric.utils import k_hop_subgraph
class IdentityHook(MessagePassing):
    def __init__(self,width):
        super().__init__(); self.in_channels=width; self.out_channels=width
    def forward(self,x,edge_index): return self.propagate(edge_index,x=x)
    def message(self,x_j): return x_j
class Adapter(torch.nn.Module):
    """Expose each custom X-THGNN layer through PyG's MessagePassing mask API."""
    def __init__(self,model,graph):
        super().__init__(); self.model=model; self.graph=graph
        width=int(model.node_encoder[0].out_features)
        self.hooks=torch.nn.ModuleList([IdentityHook(width) for _ in model.layers])
    def forward(self,x,edge_index):
        self.model.eval()  # disable dropout and batch-stat updates during explanation forwards
        g=copy.copy(self.graph); g.node_features=x; g.edge_index=edge_index
        h=self.model.node_encoder(x); importances=[]; temporal=[]; heterogeneous=[]
        src,dst=edge_index
        for i,layer in enumerate(self.model.layers):
            hook=self.hooks[i]
            hook.latest_source_embeddings=h[src]
            hook.latest_target_embeddings=h[dst]
            hook.latest_messages=h[src]
            hook(h,edge_index)
            mask=getattr(hook,'_edge_mask',None)
            if mask is not None: mask=mask.sigmoid()
            scale=getattr(hook,'message_scale',None)
            if scale is not None: mask=scale
            h,importance,ta,ha=layer(h,edge_index,g.relation_index,g.edge_features,g.timestamps,self.model.flags,edge_gate=mask)
            importances.append(importance);temporal.append(ta);heterogeneous.append(ha)
        relation_emb=self.model.relation_embedding(g.relation_index)
        edge_logits=self.model.edge_head(torch.cat([h[src],h[dst],relation_emb,g.edge_features],dim=-1)).squeeze(-1)
        campaign_repr=core.scatter_mean(0.5*(h[src]+h[dst]),g.campaign_index,len(g.campaign_ids))
        subgraph_logits=self.model.subgraph_head(campaign_repr).squeeze(-1)
        node_logits=self.model.node_head(h).squeeze(-1)
        similarity=core.F.normalize(h,dim=-1) @ core.F.normalize(self.model.prototypes,dim=-1).T / self.model.temperature
        return node_logits
def load_model(ds,seed,frame,device='cpu'):
    cp=torch.load(EXP/'runs'/ds/f'seed_{seed}'/'checkpoint.pt',map_location=device,weights_only=False); cfg=cp['config']
    b=core.EventGraphBuilder(train_fraction=float(cfg['data']['train_fraction']),validation_fraction=float(cfg['data']['validation_fraction']),feature_prefix=str(cfg['data']['feature_prefix']),history_windows_seconds=tuple(cfg['data']['history_windows_seconds']))
    tr,va,te=b.split(frame); b.fit(tr); b.fit_scaler(tr)
    probe=b.build(frame.iloc[:min(len(frame),200)].copy()); mc=cfg['model']
    model=core.XTHGNN(node_width=probe.node_features.shape[1],edge_width=probe.edge_features.shape[1],num_relations=len(b.relation_map),num_campaigns=max(1,len(b.campaign_map)),embedding_dim=int(mc['embedding_dim']),num_layers=int(mc['num_layers']),num_prototypes=int(mc['num_prototypes']),temperature=float(mc['temperature']),dropout=float(mc['dropout']),temporal_decay_init=float(mc['temporal_decay_init']),neighbor_sample_size=int(mc['neighbor_sample_size']),flags=core.AblationFlags(**cfg.get('ablation_flags',{}))).to(device)
    model.load_state_dict(cp['model_state']); model.eval()
    gs=core._closed_window_graphs(b,te,device,int(cfg['data']['temporal_consistency_window_seconds']))
    with torch.no_grad():
        best=None
        for wid,g in gs:
            out=model(g); p=out['node_logits'].sigmoid().cpu().numpy(); present=np.flatnonzero(g.node_present.cpu().numpy())
            if not len(present): continue
            indegree=np.bincount(g.edge_index[1].cpu().numpy(),minlength=len(p))
            candidates=present[indegree[present]>0]
            threshold=float(cp['thresholds']['node'])
            predicted=candidates[p[candidates]>=threshold]
            order=predicted[np.argsort(-p[predicted])] if len(predicted) else candidates[np.argsort(-p[candidates])]
            for idx in order:
                _,_,_,hop_mask=k_hop_subgraph(int(idx),num_hops=len(model.layers),edge_index=g.edge_index,num_nodes=g.node_features.size(0),relabel_nodes=False)
                if hop_mask.any():
                    alert=bool(p[idx]>=threshold)
                    candidate=(float(p[idx]),wid,g,int(idx),out['importance'].detach().cpu().numpy(),alert)
                    if best is None or candidate[0]>best[0]: best=candidate
                    break
    if best is None: raise RuntimeError(f'No test target {ds} seed {seed}')
    return model,best

def curve_metrics(model,g,target_idx,base_prob,scores,eligible):
    model.eval()  # score every method under the same inference state
    torch.manual_seed(20261117)
    e=g.edge_index.shape[1]; eligible=np.asarray(eligible,dtype=int); order=eligible[np.argsort(-np.asarray(scores)[eligible])]; fracs=np.linspace(0,1,21)
    deletion=[]; insertion=[]
    with torch.no_grad():
        zero=torch.zeros(e,device=g.edge_index.device)
        pzero=float(model(g,edge_gate=zero)['node_logits'][target_idx].sigmoid().item())
        for f in fracs:
            model.eval()
            k=int(round(f*e)); chosen=order[:k]
            keep=torch.zeros(e,device=g.edge_index.device)
            deleted=torch.ones(e,device=g.edge_index.device)
            if k:
                chosen_t=torch.as_tensor(chosen,device=keep.device); keep[chosen_t]=1; deleted[chosen_t]=0
            p_ins=float(model(g,edge_gate=keep)['node_logits'][target_idx].sigmoid().item())
            p_del=float(model(g,edge_gate=deleted)['node_logits'][target_idx].sigmoid().item())
            insertion.append(p_ins); deletion.append(p_del)
    return float(np.trapz(np.asarray(deletion),fracs)),float(np.trapz(np.asarray(insertion),fracs)),float(pzero),float(insertion[2])

def run_one(ds,seed,frame):
    model,(base_prob,wid,g,target_idx,native,target_is_alert)=load_model(ds,seed,frame)
    adapter=Adapter(model,g)
    target_cls=(model(g)['node_logits'].detach()>0).float()
    methods={}
    _,_,_,edge_mask=k_hop_subgraph(target_idx,num_hops=len(model.layers),edge_index=g.edge_index,num_nodes=g.node_features.size(0),relabel_nodes=False)
    eligible=np.flatnonzero(edge_mask.cpu().numpy())
    if not len(eligible): raise RuntimeError('No eligible message-passing neighborhood edges')
    methods['X-THGNN learned edge score']=native
    # PyG GNNExplainer optimizes a continuous edge mask against the target score.
    torch.manual_seed(seed*100+1)
    ex=Explainer(model=adapter,algorithm=GNNExplainer(epochs=30,lr=.01),explanation_type='model',model_config=ModelConfig(mode='binary_classification',task_level='node',return_type='raw'),node_mask_type=None,edge_mask_type='object')
    t=time.perf_counter(); methods['GNNExplainer']=ex(g.node_features,g.edge_index,index=target_idx).edge_mask.detach().cpu().numpy(); times={'GNNExplainer':time.perf_counter()-t}
    # Official PyG GraphMask algorithm; its shared edge gate is injected into every X-THGNN layer.
    torch.manual_seed(seed*100+2)
    ex=Explainer(model=adapter,algorithm=GraphMaskExplainer(num_layers=len(model.layers),epochs=30,lr=.01,log=False),explanation_type='model',model_config=ModelConfig(mode='binary_classification',task_level='node',return_type='raw'),node_mask_type='object',edge_mask_type='object')
    t=time.perf_counter(); methods['GraphMask (X-THGNN adapter)']=ex(g.node_features,g.edge_index,index=target_idx).edge_mask.detach().cpu().numpy(); times['GraphMask (X-THGNN adapter)']=time.perf_counter()-t
    # PGExplainer is optimized on the same target and checkpoint for 30 epochs.
    torch.manual_seed(seed*100+3)
    ex=Explainer(model=adapter,algorithm=PGExplainer(epochs=30,lr=.003),explanation_type='phenomenon',model_config=ModelConfig(mode='binary_classification',task_level='node',return_type='raw'),node_mask_type=None,edge_mask_type='object')
    t=time.perf_counter()
    candidate_nodes=torch.nonzero(g.node_present,as_tuple=False).flatten().cpu().numpy()
    candidate_nodes=candidate_nodes[candidate_nodes!=target_idx]
    with torch.no_grad(): node_prob=model(g)['node_logits'].sigmoid().cpu().numpy()
    candidate_nodes=candidate_nodes[np.argsort(node_prob[candidate_nodes])]
    pg_train_nodes=np.unique(candidate_nodes[np.linspace(0,len(candidate_nodes)-1,min(10,len(candidate_nodes)),dtype=int)])
    epoch=0
    for train_idx in pg_train_nodes:
        for _ in range(max(1,30//len(pg_train_nodes))):
            ex.algorithm.train(epoch,adapter,g.node_features,g.edge_index,target=target_cls,index=int(train_idx)); epoch+=1
    while epoch<30:
        ex.algorithm.train(epoch,adapter,g.node_features,g.edge_index,target=target_cls,index=int(pg_train_nodes[-1])); epoch+=1
    methods['PGExplainer']=ex(g.node_features,g.edge_index,target=target_cls,index=target_idx).edge_mask.detach().cpu().numpy(); times['PGExplainer']=time.perf_counter()-t
    rng=np.random.default_rng(seed*100+4); methods['Random edge ranking']=rng.random(g.edge_index.shape[1])
    rows=[]; e=g.edge_index.shape[1]; frac=.10; k=max(1,int(round(frac*len(eligible))))
    for name,scores in methods.items():
        scores=np.asarray(scores).reshape(-1)
        if len(scores)!=e: raise ValueError(f'{name} score count {len(scores)} != edges {e}')
        scores=np.where(np.isfinite(scores),scores,-1e9)
        dauc,iauc,pzero,pins=curve_metrics(model,g,target_idx,base_prob,scores,eligible)
        order=eligible[np.argsort(-scores[eligible])]; selected=torch.zeros(e,device=g.edge_index.device);selected[torch.as_tensor(order[:k],device=selected.device)]=1
        with torch.no_grad():
            p_keep=float(model(g,edge_gate=selected)['node_logits'][target_idx].sigmoid().item())
            p_del=float(model(g,edge_gate=1-selected)['node_logits'][target_idx].sigmoid().item())
        rows.append({'dataset':ds,'seed':seed,'window_id':wid,'target_node_index':target_idx,'target_is_alert':target_is_alert,'test_graph_edges':e,'eligible_khop_edges':len(eligible),'model_message_passing_layers':len(model.layers),'target_full_probability':base_prob,'method':name,'retained_fraction':k/len(eligible),'deletion_auc_probability':dauc,'insertion_auc_probability':iauc,'zero_message_probability':pzero,'insertion_probability_at_10pct':p_keep,'deletion_probability_at_10pct':p_del,'deletion_drop_at_10pct':base_prob-p_del,'fidelity_gap_at_10pct':abs(base_prob-p_keep),'runtime_seconds':times.get(name,0.0),'optimization_epochs':30 if name in ('GNNExplainer','GraphMask (X-THGNN adapter)','PGExplainer') else 0,'pg_explainer_training_targets':int(len(pg_train_nodes)) if name=='PGExplainer' else 0})
    return rows

def main():
    out=EXP/'explainer_comparison_fixed_eval';out.mkdir(parents=True,exist_ok=True)
    result_path=out/'per_target_explainer_metrics.csv'
    prior=pd.read_csv(EXP/'explainer_comparison'/'per_target_explainer_metrics.csv').drop_duplicates(['dataset','seed'])
    prior=prior[prior['eligible_khop_edges']>=5]
    records=[]
    for ds in ['cadets','theia','trace','streamspot']:
        rows0=prior[prior.dataset==ds]
        if rows0.empty: continue
        frame=core.EventGraphBuilder.load(EXP/'normalized'/f'{ds}_100000.csv')
        for _,old in rows0.iterrows():
            seed=int(old.seed)
            print(f'{ds} seed {seed} starting (candidate edges={int(old.eligible_khop_edges)})',flush=True)
            try:
                rows=run_one(ds,seed,frame)
                if any(int(r['eligible_khop_edges'])<5 for r in rows): raise RuntimeError('Target failed >=5-edge filter')
                if any(int(r['target_node_index'])!=int(old['target_node_index']) or int(r['window_id'])!=int(old['window_id']) for r in rows): raise RuntimeError('Target mismatch versus original locked target')
                for r in rows:
                    r['evaluation_mode']='model.eval: dropout disabled; batchnorm frozen'
                    r['target_edge_filter']='eligible_khop_edges >= 5'
                    r['evaluation_seed']=20261117
                records.extend(rows);pd.DataFrame(records).to_csv(result_path,index=False)
                print(f'{ds} seed {seed} done',flush=True)
            except Exception:
                traceback.print_exc(); raise
    df=pd.DataFrame(records)
    df.groupby(['dataset','method']).agg(n=('seed','count'),deletion_auc_mean=('deletion_auc_probability','mean'),deletion_auc_sd=('deletion_auc_probability','std'),insertion_auc_mean=('insertion_auc_probability','mean'),insertion_auc_sd=('insertion_auc_probability','std'),fidelity_gap_mean=('fidelity_gap_at_10pct','mean'),runtime_median=('runtime_seconds','median')).reset_index().to_csv(out/'summary_by_dataset_method.csv',index=False)
    df.groupby('method').agg(n=('seed','count'),deletion_auc_mean=('deletion_auc_probability','mean'),deletion_auc_sd=('deletion_auc_probability','std'),insertion_auc_mean=('insertion_auc_probability','mean'),insertion_auc_sd=('insertion_auc_probability','std'),fidelity_gap_mean=('fidelity_gap_at_10pct','mean'),runtime_median=('runtime_seconds','median')).reset_index().to_csv(out/'summary_overall.csv',index=False)
    print('DONE filtered fixed-evaluation targets',len(df),flush=True)
if __name__=='__main__': main()
