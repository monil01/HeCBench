"""Record owned session before exec of unchanged accepted MPI launcher."""
import os,sys,json,pathlib
p=pathlib.Path(os.environ['STUDY_PROFILE_REGISTRY'])/'session.json'
if os.getpid()!=os.getsid(0):raise SystemExit('freshsessionleaderrequired')
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w')as f:json.dump({'pid':os.getpid(),'sid':os.getsid(0),'uid':os.getuid(),'token':os.environ['STUDY_PROFILE_TOKEN']},f);f.flush();os.fsync(f.fileno())
os.execvpe(sys.argv[1],sys.argv[1:],os.environ)
