#!/usr/bin/env python3
"""CPU first-three-step trace with the full trajectory's alpha schedule."""
import argparse,pathlib,json,hashlib
import numpy as np
P=pathlib.Path(__file__).resolve().parent

def main():
 a=argparse.ArgumentParser();a.add_argument('--oracle',type=pathlib.Path,required=True);z=a.parse_args();o=z.oracle;m=json.load(open(o/'oracle_manifest.json'));source=(P/'generate_oracle.py').read_text();s=source.replace('for it in range(1,iterations+1):','for it in range(1,min(iterations,3)+1):').replace('if it==1:trace=', 'if it==3:trace=');assert s!=source and 'if it==3:trace='in s
 ns={'__file__':str(P/'generate_oracle.py'),'__name__':'cpu_oracle_trace3_module'};exec(compile(s,'<source-bound-CPU-three-step-trace>','exec'),ns)
 image=np.fromfile(o/'input_img.bin',dtype='<c16').astype('<c8');mask=np.fromfile(o/'input_mask.bin',dtype='<c16').astype('<c8');out,trace=ns['oracle'](image,mask,tuple(m['shape']),m['iterations']);rows={}
 for k,v in trace.items():
  path=o/('trace_iteration3_'+k+'.bin');a=np.asarray(v,dtype='<c16').ravel();a.tofile(path);rows[k]={'path':str(path),'dtype':'complex128','count':a.size,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 proof={'source_script':str(P/'generate_oracle.py'),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'explicit_CPU_only_transformations':['Execute steps1..3 only, retaining requested fulliterations alpha denominator.','Capture third-iteration arrays instead of first.'],'transformed_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'helper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'actual_executed_steps':3,'alpha_schedule_total_iterations':m['iterations'],'full_trajectory_claim':False,'trace_arrays':rows};(o/'trace_iteration3_manifest.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'actual_CPU_trace_steps':3,'arrays':len(rows),'output':str(o)}))
if __name__=='__main__':main()
