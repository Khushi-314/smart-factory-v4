import pandas as pd
from simulator import run_four, weibull_third

params=pd.read_csv('data/processing_params.csv')
q={p:2 for p in ['PR1','PR2','PR3','PR4']}
b={f'MC{i}':0 for i in range(1,10)}
rel={}
for i in range(1,10):
    w=weibull_third('MTTF + beta',5.0,1.5)
    rel[f'MC{i}']={**w,'repair_mean_min':5.0,'repair_cv':0.30}
res=run_four(params,q,1,b,rel,42,False)
assert all(r['completed']==8 for r in res.values())
assert abs(res['No failure']['end_time_s']-res['Instant repair']['end_time_s']) < 1e-6 or res['Instant repair']['failures']>0
print('basic sanity check passed')
