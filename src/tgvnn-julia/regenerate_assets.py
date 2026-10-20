#!/usr/bin/env python3
"""Cold CPU-only recipes; exact hashes checked, no GPU execution or package installation."""
import argparse,pathlib,json,hashlib,subprocess,sys,os,shutil,time
P=pathlib.Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--proof',type=pathlib.Path);a.add_argument('--resume-existing-oracle',action='store_true');z=a.parse_args();o=z.output.resolve();o.mkdir(parents=True,exist_ok=True);expected=json.load(open(P/'asset_hashes.json'));env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='8';env['OMP_NUM_THREADS']='8';start=time.monotonic();commands=[];rows=[]
 for case,it in [('pincat',500),('rectangular',7)]:
  dest=o/case
  for script,args in [('generate_oracle.py',['--output',str(dest),'--case',case,'--iterations',str(it)]),('generate_trace3.py',['--oracle',str(dest)]),('prepare_oracle_assets.py',['--oracle',str(dest)])]:
   cmd=[sys.executable,str(P/script),*args]
   if z.resume_existing_oracle and script=='generate_oracle.py' and (dest/'oracle_manifest.json').exists():
    commands.append({'reused_completed_CPU_generation':str(dest/'oracle_manifest.json'),'scope':'Only resume local completed oracle; all92 binary hashes still mandatory.'});continue
   phase_env=env.copy()
   if script=='prepare_oracle_assets.py':
    phase_env.pop('OPENBLAS_NUM_THREADS',None);phase_env.pop('OMP_NUM_THREADS',None)
   commands.append({'command':cmd,'BLAS_threads':'library_default_unset'if script=='prepare_oracle_assets.py'else 8});subprocess.run(cmd,env=phase_env,check=True)
  for name,meta in expected['cases'][case]['files'].items():
   f=dest/name;got={'case':case,'name':name,'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)};assert got['bytes']==meta['bytes']and got['sha256']==meta['sha256'],('CPU regeneration different bytes',got,meta);rows.append(got)
 cmd=['g++','-O3','-std=c++17','-shared','-fPIC',str(P/'vendor_bridge.cpp'),'-lcufft','-lcublas','-lcusolver','-lcudart','-ldl','-o',str(o/'libtgv_vendor.so')];commands.append(cmd);subprocess.run(cmd,check=True);f=o/'libtgv_vendor.so';assert f.stat().st_size==expected['host_library']['bytes']and sha(f)==expected['host_library']['sha256'],'Hostlibrarydifferentbytes/compiler; inspectenvironment, never silentlyreplace.'
 target=P/'libtgv_vendor.so'
 if target.exists():assert sha(target)==sha(f),'Existingmeasuredlibrarydiffers; willnotoverwrite.'
 else:shutil.copyfile(f,target)
 proof={'status':'byte_exact','CPU_only':True,'GPU_execution':False,'source_inputs':'Trackedoriginalsrc/tgvnn-cuda/data/pincat.ra+pincat_mask.ra, strictRAreader; deterministicrectangularcomplexfixturefromgenerator. No downloaded/untrackedirreplaceableasset.','files':rows,'files_count':len(rows),'bytes':sum(x['bytes']for x in rows),'host_library':{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size},'commands':commands,'wall_seconds':time.monotonic()-start,'OPENBLAS_NUM_THREADS':8,'OMP_NUM_THREADS':8,'scalar_SER_generation_threads':'original library default with OPENBLAS_NUM_THREADS and OMP_NUM_THREADS unset; array evolution uses8','resumed_existing_CPU_oracle':z.resume_existing_oracle,'python':sys.version,'source_recipes':{str(P/x):sha(P/x)for x in ['generate_oracle.py','generate_trace3.py','prepare_oracle_assets.py','vendor_bridge.cpp','asset_hashes.json']},'irreplaceable_assets':[],'scope':'ExactCPUfixture/oracle/library regeneration onpinnedNumPy/BLAS/compilerenvironment. No freshcoldGPUverification or recoveryofpreviousmissingrawGPUhistory.'};out=z.proof or(o/'regeneration_proof.json');out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':'byte_exact','files':len(rows),'bytes':proof['bytes'],'proof':str(out)}))
if __name__=='__main__':main()
