"""Post-hoc THEIA threshold sensitivity; uses saved held-out predictions only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs'/'theia'
DELTAS=[0.0,0.02,0.05,0.075,0.10,0.15,0.20]
rows=[]
for seed in [11,22,33,44,55,66,77,88,99,110]:
    run=RUNS/f'seed_{seed}'
    met=json.loads((run/'metrics.json').read_text())
    threshold=float(met['node']['threshold'])
    pred=pd.read_csv(run/'node_predictions.csv')
    y=pred['label'].to_numpy(dtype=int)
    p=pred['probability'].to_numpy(dtype=float)
    for delta in DELTAS:
        used=min(1.0,threshold+delta)
        yh=(p>=used).astype(int)
        rows.append({'seed':seed,'threshold_increment':delta,'validation_selected_threshold':threshold,
                     'operating_threshold':used,'test_f1':f1_score(y,yh,zero_division=0),
                     'test_precision':precision_score(y,yh,zero_division=0),
                     'test_recall':recall_score(y,yh,zero_division=0),
                     'has_alert':bool(yh.any()),'test_auroc':met['node']['auroc'],
                     'test_auprc':met['node']['auprc']})
raw=pd.DataFrame(rows)
(ROOT/'summaries').mkdir(parents=True,exist_ok=True)
raw.to_csv(ROOT/'summaries'/'theia_threshold_sensitivity_seed_metrics.csv',index=False)
summary=[]
for delta,g in raw.groupby('threshold_increment',sort=True):
    k=int(g['has_alert'].sum())
    summary.append({'threshold_increment':delta,'alert_producing_seeds':f'{k}/10',
                    'alert_pair_support':f'{k*(k-1)//2}/45',
                    'mean_test_f1':g.test_f1.mean(),'sd_test_f1':g.test_f1.std(ddof=1),
                    'mean_test_precision':g.test_precision.mean(),'mean_test_recall':g.test_recall.mean(),
                    'mean_test_auroc':g.test_auroc.mean(),'mean_test_auprc':g.test_auprc.mean()})
pd.DataFrame(summary).to_csv(ROOT/'summaries'/'theia_threshold_sensitivity_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
