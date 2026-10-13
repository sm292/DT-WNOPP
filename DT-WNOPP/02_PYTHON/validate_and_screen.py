from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'03_DATA'/'processed_measurements.csv'; OUT=ROOT/'04_RESULTS'
TARGETS={'throughput_observed_source_units':'max','jitter_observed_ms':'min','packet_loss_observed_source_units':'min'}
FEATURES=['resource_fraction','offered_downlink_source_units']

def run(seed=42,repeats=20,latency_limit=None,loss_limit=None):
    df=pd.read_csv(DATA); required=['scenario',*TARGETS,*FEATURES]
    missing=[c for c in required if c not in df]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    for c in [*TARGETS,*FEATURES]: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=['scenario',*TARGETS]).reset_index(drop=True)
    df['config_group']=df.scenario.astype(str).str.replace(r'_rep\d+$','',regex=True,case=False)
    xcols=[c for c in FEATURES if df[c].notna().any() and df[c].nunique(dropna=True)>1]
    groups=df.config_group
    if groups.nunique()<5: raise ValueError('Need at least five distinct configuration groups.')
    models={'Mean baseline':None,'Ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1.0)),
      'Random Forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=300,min_samples_leaf=3,random_state=seed,n_jobs=1)),
      'Extra Trees':make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,random_state=seed,n_jobs=1)),
      'Gradient Boosting':make_pipeline(SimpleImputer(strategy='median'),GradientBoostingRegressor(n_estimators=150, max_depth=2, random_state=seed))}
    records=[]; predictions=[]; splitlog=[]
    X=df[xcols].apply(pd.to_numeric,errors='coerce')
    for rep in range(repeats):
      a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+rep).split(X,groups=groups))
      splitlog.append({'repeat':rep,'seed':seed+rep,'train_groups':int(groups.iloc[a].nunique()),'test_groups':int(groups.iloc[b].nunique())})
      for target in TARGETS:
        usable_train=a[df[target].iloc[a].notna().to_numpy()]; usable_test=b[df[target].iloc[b].notna().to_numpy()]
        if not len(usable_train) or not len(usable_test): continue
        ytr=df[target].iloc[usable_train].astype(float); yte=df[target].iloc[usable_test].astype(float)
        for name,model in models.items():
          fitted=model
          if model is None: pred=np.full(len(yte),ytr.mean())
          else:
            from sklearn.base import clone
            fitted=clone(model); fitted.fit(X.iloc[usable_train],ytr); pred=fitted.predict(X.iloc[usable_test])
          records.append({'repeat':rep,'target':target,'model':name,'MAE':mean_absolute_error(yte,pred),'RMSE':np.sqrt(mean_squared_error(yte,pred)),'R2':r2_score(yte,pred) if len(yte)>1 else np.nan,'n_train':len(ytr),'n_test':len(yte)})
          if rep==0: predictions.extend({'repeat':rep,'scenario':df.scenario.iloc[idx],'target':target,'model':name,'actual':float(actual),'predicted':float(p)} for idx,actual,p in zip(usable_test,yte,pred))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(OUT/'validation_metrics_repeated.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'validation_predictions.csv',index=False)
    pd.DataFrame(splitlog).to_csv(OUT/'validation_splits.csv',index=False)
    # Explicit QoS thresholds are optional and never inferred from observed minima/maxima.
    scored=df.copy(); scored['qos_eligible']=True
    if latency_limit is not None: scored['qos_eligible'] &= scored.jitter_observed_ms<=latency_limit
    if loss_limit is not None: scored['qos_eligible'] &= scored.packet_loss_observed_source_units<=loss_limit
    objective=scored[list(TARGETS)].to_numpy(dtype=float,copy=True)
    objective[:,1:]*=-1  # maximise throughput while minimising jitter and loss
    complete=np.isfinite(objective).all(axis=1)
    feasible=scored.qos_eligible.to_numpy() & complete
    efficient=np.zeros(len(scored),dtype=bool)
    for i in np.flatnonzero(feasible):
      peers=objective[np.flatnonzero(feasible)]
      efficient[i]=not np.any(np.all(peers>=objective[i],axis=1)&np.any(peers>objective[i],axis=1))
    scored['pareto_efficient_observation']=efficient
    scored['candidate_status']=np.select(
      [~complete,~scored.qos_eligible,scored.pareto_efficient_observation],
      ['incomplete_outcomes_not_ranked','fails_configured_qos','feasible_observed_pareto_candidate'],
      default='feasible_observed_dominated_candidate')
    if latency_limit is None and loss_limit is None:
      scored['candidate_status']=scored['candidate_status'].replace({
        'feasible_observed_pareto_candidate':'observed_pareto_only_qos_limits_not_configured',
        'feasible_observed_dominated_candidate':'observed_dominated_only_qos_limits_not_configured'})
    scored.to_csv(OUT/'qos_candidate_screen.csv',index=False)
    # Matched scenario summaries at available allocation levels; no extrapolated allocations.
    scenario=df.assign(config_group=groups).groupby(['config_group','resource_fraction'],dropna=False).agg(
      n=('scenario','size'),throughput=('throughput_observed_source_units','mean'),jitter=('jitter_observed_ms','mean'),loss=('packet_loss_observed_source_units','mean')).reset_index()
    scenario.to_csv(OUT/'matched_resource_scenario_summary.csv',index=False)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'repeats':repeats,'features':xcols,'targets':TARGETS,'qos_limits':{'jitter_ms':latency_limit,'packet_loss_source_units':loss_limit},'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__,'source_csv':str(DATA),'source_sha256':__import__('hashlib').sha256(DATA.read_bytes()).hexdigest(),'allocation_values':sorted(df.resource_fraction.dropna().unique().tolist()),'claim_limit':'Grouped holdout predictive comparison; observed resource screen only. No causal effect or global optimum established.'}
    (OUT/'validation_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(records).groupby(['target','model'])[['MAE','RMSE','R2']].mean().round(4).to_string())
    print(f'Wrote reproducible validation and screening outputs to {OUT}')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); p.add_argument('--repeats',type=int,default=20); p.add_argument('--max-jitter-ms',type=float); p.add_argument('--max-packet-loss',type=float); a=p.parse_args(); run(a.seed,a.repeats,a.max_jitter_ms,a.max_packet_loss)
