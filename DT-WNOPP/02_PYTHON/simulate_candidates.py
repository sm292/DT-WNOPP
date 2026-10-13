from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def matlab_quote(value: str) -> str:
    return "'"+value.replace("'","''")+"'"

def main():
    p=argparse.ArgumentParser(description='Run matched Simulink proxy cases at user-selected resource fractions.')
    p.add_argument('--matlab-exe', help='Full path to matlab.exe; searches Program Files if omitted.')
    p.add_argument('--allocations', nargs='+', type=float, required=True, help='Resource fractions, each in (0,1].')
    a=p.parse_args()
    if any(not 0<x<=1 for x in a.allocations): p.error('Each allocation must be greater than 0 and at most 1.')
    inp=ROOT/'03_DATA'/'processed_measurements.csv'; out=ROOT/'04_RESULTS'/'candidate_simulations'; out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(inp)
    for col in ('time_step','traffic_load','resource_fraction'):
        if col not in data: raise ValueError(f'Missing required input column {col}')
    matlab=Path(a.matlab_exe) if a.matlab_exe else None
    if matlab is None:
        found=sorted(Path(os.environ.get('ProgramFiles',r'C:\Program Files')).glob('MATLAB/R*/bin/matlab.exe'),reverse=True)
        if not found: raise FileNotFoundError('MATLAB was not found; pass --matlab-exe.')
        matlab=found[0]
    if not matlab.is_file(): raise FileNotFoundError(matlab)
    matlab_dir=ROOT/'01_MATLAB'; runner=matlab_dir/'run_dt_wnopp_twin.m'
    combined=[]; provenance=[]
    for allocation in sorted(set(a.allocations)):
        label=f'{allocation:.6g}'.replace('.','p')
        candidate=out/f'input_resource_{label}.csv'; output=out/f'proxy_resource_{label}.csv'
        data.assign(resource_fraction=allocation).to_csv(candidate,index=False)
        expr=f"addpath({matlab_quote(str(matlab_dir))}); run_dt_wnopp_twin({matlab_quote(str(candidate))},{matlab_quote(str(output))});"
        completed=subprocess.run([str(matlab),'-batch',expr],cwd=ROOT,text=True,capture_output=True)
        provenance.append({'resource_fraction':allocation,'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr})
        if completed.returncode: raise RuntimeError(f'MATLAB failed at allocation {allocation}: {completed.stderr or completed.stdout}')
        if not output.is_file(): raise RuntimeError(f'MATLAB did not write {output}')
        result=pd.read_csv(output); result['resource_fraction']=allocation; combined.append(result)
    proxies=pd.concat(combined,ignore_index=True)
    proxies.to_csv(out/'matched_proxy_outputs.csv',index=False)
    baseline=float(min(set(a.allocations)))
    metric_columns=['latency_proxy_ms','throughput_proxy_normalised','packet_loss_proxy_pct']
    base=proxies[proxies.resource_fraction==baseline].set_index('simulation_time')[metric_columns]
    deltas=[]
    for allocation in sorted(set(a.allocations)):
        candidate=proxies[proxies.resource_fraction==allocation].set_index('simulation_time')[metric_columns]
        paired=base.join(candidate,lsuffix='_baseline',rsuffix='_candidate',how='inner')
        for metric in metric_columns:
            change=paired[f'{metric}_candidate']-paired[f'{metric}_baseline']
            denom=paired[f'{metric}_baseline'].replace(0,float('nan'))
            deltas.append({'baseline_fraction':baseline,'candidate_fraction':allocation,'metric':metric,'matched_samples':len(paired),'baseline_mean':paired[f'{metric}_baseline'].mean(),'candidate_mean':paired[f'{metric}_candidate'].mean(),'mean_absolute_change':change.mean(),'mean_percent_change':(100*change/denom).mean()})
    pd.DataFrame(deltas).to_csv(out/'matched_proxy_deltas.csv',index=False)
    summary=proxies.groupby('resource_fraction').agg(
      n=('simulation_time','size'),latency_proxy_ms_mean=('latency_proxy_ms','mean'),
      throughput_proxy_normalised_mean=('throughput_proxy_normalised','mean'),
      packet_loss_proxy_pct_mean=('packet_loss_proxy_pct','mean')).reset_index()
    summary.to_csv(out/'matched_proxy_summary.csv',index=False)
    (out/'matlab_run_log.json').write_text(json.dumps({'matlab_executable':str(matlab),'matlab_release_folder':matlab.parent.parent.name,'input':str(inp),'input_rows':len(data),'baseline_fraction':baseline,'allocation_fractions':sorted(set(a.allocations)),'runs':provenance,'interpretation':'Paired Simulink proxy comparison under shared traffic input; not independently measured network outcomes and not proof of a real-world optimum.'},indent=2),encoding='utf-8')
    print(summary.to_string(index=False)); print(f'Proxy results saved under {out}')
if __name__=='__main__': main()
