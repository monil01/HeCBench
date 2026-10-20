from pathlib import Path
import fcntl,subprocess,os,json,time
P=Path(__file__).resolve().parent
cmd=[str(P/'original_cli'),'--deck',str(P.parent/'original_assets/data/bm1'),'--wgsize','256','--iterations','8']
with open('/tmp/hecbench-julia-study-gpu.lock','w')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);start=time.time();q=subprocess.run(cmd,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='6'),timeout=300)
(P/'original_cli.log').write_text(q.stdout+q.stderr);(P/'original_cli_receipt.json').write_text(json.dumps({'command':cmd,'exit_code':q.returncode,'wall_seconds':time.time()-start,'source_GPU_body_unchanged':True,'scope':'OriginalfullBM1/65536poses/default8iterationsreadiness; checker divides by signed reference energy and skips small energies, so not independent accuracy. Timer lacks completion fence afterwarmup; not acceptedfive-trialmean.'},indent=2)+'\n');print(q.stdout);q.check_returncode()
