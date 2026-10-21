"""Register the exact target PID before exec/CUDA initialization. No GPU queries."""
import os,json,sys,pathlib
P=pathlib.Path
rank=int(os.environ['OMPI_COMM_WORLD_RANK']);local=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
if rank!=local or rank not in(0,1)or os.environ['OMPI_COMM_WORLD_SIZE']!='2' or os.environ['CUDA_VISIBLE_DEVICES']!='5,6':raise SystemExit('exact2samehostranks/visibilityrequired')
root=P(os.environ['STUDY_PROFILE_REGISTRY']);session=json.loads((root/'session.json').read_text())
if os.getsid(0)!=session['sid']or session['token']!=os.environ['STUDY_PROFILE_TOKEN']:raise SystemExit('rank must remain in fresh owned session')
stat=(P('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split();record={'rank':rank,'pid':os.getpid(),'sid':os.getsid(0),'uid':os.getuid(),'starttime':stat[19],'token':os.environ['STUDY_PROFILE_TOKEN'],'target_argv':sys.argv[1:]}
record['effective_environment']={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','OPAL_PREFIX','UCX_TLS','UCX_MEMTYPE_CACHE','OMPI_MCA_coll','UCX_ERROR_SIGNALS','JULIA_CUDA_MEMORY_POOL','TMPDIR','CUDA_CACHE_PATH','XDG_CACHE_HOME','JULIA_DEPOT_PATH','OMPI_MCA_orte_tmpdir_base','UCX_POSIX_DIR','OMPI_COMM_WORLD_SIZE','OMPI_COMM_WORLD_RANK','OMPI_COMM_WORLD_LOCAL_RANK']}
fd=os.open(root/f'rank{rank}.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w')as f:json.dump(record,f);f.flush();os.fsync(f.fileno())
if len(sys.argv)<2:raise SystemExit('targetargvrequired')
os.execvpe(sys.argv[1],sys.argv[1:],os.environ)
