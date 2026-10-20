#!/usr/bin/env python3
"""Cold CPU input/oracle regeneration, complete binary hashes, no GPU or installs."""
import pathlib,json,hashlib,argparse,subprocess,sys,os,time
P=pathlib.Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--proof',type=pathlib.Path,required=True);z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);expected=json.load(open(P/'asset_hashes.json'));env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='8';env['OMP_NUM_THREADS']='8';start=time.monotonic();rows=[];commands=[]
 for case,meta in expected['cases'].items():
  folder=z.output/case;cmd=[sys.executable,str(P/'generate_oracle.py'),'--output',str(folder),'--iterations',str(meta['iterations'])];commands.append(cmd);subprocess.run(cmd,env=env,check=True)
  for name,want in meta['files'].items():
   f=folder/name;assert f.stat().st_size==want['bytes']and sha(f)==want['sha256'],(case,name,'differentCPUbytes');rows.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)})
 cmd=[sys.executable,str(P/'prepare_fourier_metric.py'),'--oracle',str(z.output/'i1'),'--output',str(z.output/'Fourier_metric_units_regenerated.json')];commands.append(cmd);subprocess.run(cmd,env=env,check=True);new=json.load(open(z.output/'Fourier_metric_units_regenerated.json'));old=json.load(open(P/'Fourier_metric_units.json'));assert new['fields']==old['fields'],'FourierCPUunits changed';proof={'status':'all62_CPU_binary_assets_byte_exact','files':rows,'files_count':len(rows),'bytes':sum(r['bytes']for r in rows),'Fourier_units_substantive_fields_exact':True,'volatile_manifest_runtime_metadata_byteidentity_not_claimed':True,'commands':commands,'elapsed_seconds':time.monotonic()-start,'GPU_execution':False,'package_or_OS_installation':False,'scope':'PinnedNumPy2.5.4/currentCPUenvironment; allgeneratedbinarybytes exact; no freshcoldGPUvalidation or restorationoflostoldsidecars.'};z.proof.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':proof['status'],'files':len(rows),'bytes':proof['bytes']}))
if __name__=='__main__':main()
