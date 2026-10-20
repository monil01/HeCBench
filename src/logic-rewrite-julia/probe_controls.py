#!/usr/bin/env python3
"""Native-readiness diagnostics, not the five-trial measurement harness."""
import argparse,hashlib,json,pathlib,subprocess,time
from aiger_controls import PASSES,evaluate
p=argparse.ArgumentParser();p.add_argument('--source-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2]);p.add_argument('--output-dir',type=pathlib.Path,required=True);p.add_argument('--repeats',type=int,default=2);p.add_argument('--cut-size',type=int,default=12);a=p.parse_args();root=a.source_root.resolve();parent=root/'src/logic-rewrite-julia';out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);records=[]
for n in (4,6,9,12):
    src=parent/'controls'/f'control_pi{n}.aig';base,expected=evaluate(src)
    for rep in range(a.repeats):
        folder=out/f'pi{n}_repeat{rep}';folder.mkdir(exist_ok=True);paths=[folder/f'pass{k:02d}.aig' for k in range(1,16)];commands=[f'read {src}']
        for op,dest in zip(PASSES,paths):
            if op.startswith('refactor'):op+=f' -K {a.cut_size}'
            commands.extend((op,f'write {dest}'))
        commands+=['time'];cmd=['flock','-x','/tmp/hecbench-julia-study-gpu.lock','env','CUDA_VISIBLE_DEVICES=6','timeout','300',str(parent/'reference_history/main'),'-c','; '.join(commands)]
        start=time.perf_counter()
        with open(folder/'native.log','w') as log:r=subprocess.run(cmd,cwd=parent/'reference_history',stdout=log,stderr=subprocess.STDOUT)
        record=dict(cut_size=a.cut_size,primary_inputs=n,repeat=rep,returncode=r.returncode,wall_seconds=time.perf_counter()-start,diagnostic_only=True,input=base,passes=[])
        for k,q in enumerate(paths,1):
            if q.exists():
                metrics,truth=evaluate(q);record['passes'].append(dict(pass_index=k,operation=PASSES[k-1],full_truth_equivalent=truth==expected,**metrics))
        record['complete_boolean_pass']=r.returncode==0 and len(record['passes'])==15 and all(x['full_truth_equivalent'] for x in record['passes']);records.append(record)
        tmp=out/'readiness.tmp';tmp.write_text(json.dumps(records,indent=2)+'\n');tmp.replace(out/'readiness.json');print(json.dumps({k:v for k,v in record.items() if k not in ('passes','input')}),flush=True)
        if not record['complete_boolean_pass']:raise SystemExit(1)
print('All native full-pipeline Boolean checks passed.',flush=True)
