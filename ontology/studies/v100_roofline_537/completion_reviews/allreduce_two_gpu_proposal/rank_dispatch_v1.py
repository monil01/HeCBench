"""NCU wraps rank1 only; rank0 runs the same accepted worker command."""
import os,sys,json,pathlib
P=pathlib.Path
plan=json.loads(P(sys.argv[1]).read_text());rank=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']);registry=P(os.environ['STUDY_PROFILE_REGISTRY'])
if os.environ['OMPI_COMM_WORLD_SIZE']!='2'or rank not in(0,1):raise SystemExit('exact2ranksrequired')
worker=[sys.executable,plan['rank_worker'],*plan['rank_target_argv']]
if rank==0:os.execvpe(worker[0],worker,os.environ)
stat=(P('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split();record={'role':'profiler','rank':1,'pid':os.getpid(),'sid':os.getsid(0),'uid':os.getuid(),'starttime':stat[19],'token':os.environ['STUDY_PROFILE_TOKEN'],'executable':plan['ncu_executable']}
fd=os.open(registry/'profiler.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w')as f:json.dump(record,f);f.flush();os.fsync(f.fileno())
cmd=[plan['ncu_executable'],*plan['ncu_arguments'],*worker];os.execvpe(cmd[0],cmd,os.environ)
