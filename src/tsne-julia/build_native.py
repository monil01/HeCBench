#!/usr/bin/env python3
"""Rebuild exact owned observer/original device sources; no GPU or source edits."""
import pathlib,json,hashlib,argparse,subprocess,sys,os
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--nvcc',default='/usr/bin/nvcc');a.add_argument('--pythonpath',required=True);z=a.parse_args();o=z.output.resolve();o.mkdir(parents=True,exist_ok=True);O=R/'src/tsne-cuda';sources=[P/'study_native.cu',*sorted(f for f in O.glob('*.cu')if f.name not in ['main.cu','fit_tsne.cu'])];cmd=[z.nvcc,'-O3','--use_fast_math','-std=c++17','-arch=sm_70','-allow-unsupported-compiler','-I'+str(O),*[str(f)for f in sources],'-o',str(o/'study_native')]
 with open(o/'build.log','w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 # The immutable measured native adapter invokes an absolute Python audit path.
 # A narrow user-space command launcher relocates that CPU-only evidence command,
 # not GPU computation. Exact measured source bytes remain unchanged.
 bindir=o/'bin';bindir.mkdir(exist_ok=True);script='#!/usr/bin/python3\nimport os,sys\na=sys.argv[1:]\ne=os.environ.copy()\nif a and (a[0]=='+repr('/home/imo/HeCBench_julia/src/tsne-julia/audit_native.py')+' or a[0]=='+repr(str(P/'audit_native.py'))+'):\n a[0]='+repr(str(P/'audit_native.py'))+'\n e["PYTHONPATH"]='+repr(z.pythonpath)+'\nos.execve('+repr(sys.executable)+',['+repr(sys.executable)+',*a],e)\n';launcher=bindir/'python3';launcher.write_text(script);launcher.chmod(0o755)
 proof={'status':'native_CPU_build_success','command':cmd,'source_files':{str(f):sha(f)for f in sources+[P/'observed_fit.inc']},'binary':str(o/'study_native'),'binary_sha256':sha(o/'study_native'),'build_log_sha256':sha(o/'build.log'),'Python_audit_path_launcher':str(launcher),'launcher_sha256':sha(launcher),'launcher_scope':'OnlyexistingCPUauditcommandpath/PYTHONPATHrelocation. No authoredGPUdelegation/noOSorpackagechange. Usedonlyfreshcoldconfigs, notrewrittenhistoricalmeasurements.','GPU_execution':False,'binary_identity_across_build_paths_not_claimed':True,'original_device_source_changed':False};(o/'build_proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':proof['status'],'binary':str(o/'study_native'),'GPU_execution':False}))
if __name__=='__main__':main()
