from pathlib import Path
import subprocess,json,os,hashlib
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;out=r/'.porting-state/porting_logs/blas-gemmEx2-julia-shared';exe=out/'probe_original';cmd=['nvcc','-O3','-std=c++17','-arch=sm_70','--cudart','shared','-Xlinker','-rpath','-Xlinker','/usr/lib/x86_64-linux-gnu',str(p/'probe_original.cu'),'-lcublasLt','-ldl','-o',str(exe)];z=subprocess.run(cmd,cwd=r,capture_output=True,text=True);(out/'probe_build.log').write_text(z.stdout+z.stderr);print('BUILD',z.returncode,flush=True)
if z.returncode:raise SystemExit(z.returncode)
rows=[]
for i,name in enumerate(['fp64','fp32_default','fp32_label_fp16','fp32_label_bf16','fp32_label_tf32','fp16_compute16','fp16_compute32','bf16_compute32','int8_compute32']):
 z=subprocess.run([str(exe),str(i)],cwd=r,capture_output=True,text=True);f=out/f'original_{name}.log';f.write_text(z.stdout+z.stderr);rows.append(dict(path=str(f.relative_to(r)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),kind=name,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr));print(name,z.returncode,z.stdout[-260:],flush=True)
(out/'original_probes.json').write_text(json.dumps(dict(source=str((p/'probe_original.cu').relative_to(r)),source_sha256=hashlib.sha256((p/'probe_original.cu').read_bytes()).hexdigest(),build=cmd,rows=rows),indent=2)+'\n')
