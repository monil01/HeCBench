#!/usr/bin/env python3
"""Build an independent cold protocol configuration; does not execute GPUs."""
import pathlib,argparse,json,subprocess,sys,shutil
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('--build',type=pathlib.Path,required=True);a.add_argument('--captures',type=pathlib.Path,required=True);a.add_argument('--julia',default=shutil.which('julia'));a.add_argument('--out-config',type=pathlib.Path,required=True);z=a.parse_args();B=z.build.resolve();C=z.captures.resolve();assert not C.exists()and not B.exists()and not z.out_config.exists(),'Refuse to overwrite previous capture/build/protocol';assert z.julia,'Provide --julia path' 
# Existing pinned GSL dependency must first be reproduced by fetch_build_gsl.py.
subprocess.run([sys.executable,str(P/'build_study.py'),str(B)],check=True)
c=json.loads((P/'study_config_jacc.json').read_text());oldroot=str(pathlib.Path(c['models']['cuda']['cwd']).parents[1]);oldbuild='/tmp/hecbench-portable-infrastructure-sss-build03';oldcaptures='/tmp/hecbench-portable-infrastructure-sss-captures'
def rewrite(x):
 if isinstance(x,str):return x.replace(oldroot,str(R)).replace(oldbuild,str(B)).replace(oldcaptures,str(C))
 if isinstance(x,list):return[rewrite(v)for v in x]
 if isinstance(x,dict):return{k:rewrite(v)for k,v in x.items()}
 return x
c=rewrite(c)
for model in ('cuda_jl','ka','jacc'):c['models'][model]['command'][0]=str(pathlib.Path(z.julia).resolve())
for k in ['reuse_result','reuse_models']:c.pop(k,None)
c['configuration_id']=c['configuration_id']+'_cold_reproduction';z.out_config.parent.mkdir(parents=True,exist_ok=True);z.out_config.write_text(json.dumps(c,indent=2)+'\n')
print('Prepared fresh all-four protocol; no GPU commands executed. Review then invoke:')
print(sys.executable,str(R/'tools/verify_portable_julia.py'),'sss','--config',str(z.out_config.resolve()))
