"""CPU metadata/source-copy preparation only; never invokes GPU/build/runner."""
from pathlib import Path
import json,hashlib,copy,shutil,re,datetime
P=Path;O=P(__file__).resolve().parent;W=P('/home/imo/HeCBench_julia');SRC=W/'src/allreduce-julia';BASE=P('/tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512/allreduce');H=P('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1')
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def bind(p):p=P(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
envreview=SRC/'environment_review.json';physical=json.loads(envreview.read_text())['physical_devices'];expected={'5':'GPU-391e8985-ae24-d6b8-b478-e0d26dfe6df5','6':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'}
if physical!=expected:raise ValueError('RecordedUUIDmetadatachanged')
(O/'prepared').mkdir(exist_ok=False);models={};proofs={};allbindings={}
common=SRC/'study_common.jl';original=common.read_text();text=original;changes=[]
def replace(before,after):
 global text
 if text.count(before)!=1:raise ValueError('Exacthostpatchoccurrence '+before)
 text=text.replace(before,after,1);changes.append({'before':before,'after':after})
replace('using CUDA,Libdl','using CUDA,Libdl\nconst PROFILE_LIBRARY=Libdl.dlopen(joinpath(pwd(),"profile_window.so"))\nconst PROFILE_SET=Libdl.dlsym(PROFILE_LIBRARY,:profile_set_trial)\nconst PROFILE_BEFORE=Libdl.dlsym(PROFILE_LIBRARY,:profile_before_add)\nconst PROFILE_AFTER=Libdl.dlsym(PROFILE_LIBRARY,:profile_after_add)\nconst PROFILE_FINISH=Libdl.dlsym(PROFILE_LIBRARY,:profile_finish)')
replace(' reduce(c,starts[recv+1],lens[recv+1]);CUDA.device_synchronize(blocking=true)',' ccall(PROFILE_BEFORE,Cvoid,(Cint,Cint,Cint,Cint),c.n,rank,starts[recv+1],lens[recv+1])\n reduce(c,starts[recv+1],lens[recv+1]);CUDA.device_synchronize(blocking=true)\n ccall(PROFILE_AFTER,Cvoid,())')
replace(' for trial in 1:5\n',' for trial in 1:5\n  ccall(PROFILE_SET,Cvoid,(Cint,),trial)\n')
replace(' ccall(MS["finish"],Cvoid,())',' ccall(PROFILE_FINISH,Cvoid,(Cint,),rank)\n ccall(MS["finish"],Cvoid,())')
inverse=text
for q in reversed(changes):inverse=inverse.replace(q['after'],q['before'],1)
if inverse!=original:raise ValueError('Commonhostpatchinverse')
commoncopy=O/'prepared/study_common.jl';commoncopy.write_text(text);commonproof={'original':bind(common),'copy':bind(commoncopy),'inverse_exact':True,'patches':changes}
metrics=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','dram__bytes.sum','dram__bytes.sum.peak_sustained','dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__t_bytes.sum','lts__t_bytes.sum.peak_sustained','lts__cycles_elapsed.avg.per_second','smsp__sass_thread_inst_executed_op_integer_pred_on.sum']+['smsp__sass_thread_inst_executed_op_'+x+'_pred_on.sum'for x in ['hadd','hmul','hfma','fadd','fmul','ffma','dadd','dmul','dfma']]+['sm__sass_thread_inst_executed_op_'+x+'_pred_on.sum.peak_sustained'for x in ['hfma','ffma','dfma']]+['device__attribute_pci_'+x+'_id'for x in ['domain','bus','device']]
ncu=P(shutil.which('ncu')).resolve();nvcc=P(shutil.which('nvcc')).resolve();smi=P(shutil.which('nvidia-smi')).resolve();python=P(shutil.which('python3')).resolve();guard=W/'tools/v100_profile_write_guard.py'
libs=[H/'ompi/lib/libmpi.so',H/'ucx/lib/libucp.so',H/'ucx/lib/libuct.so',H/'ucx/lib/libucs.so']
for m in ['cuda','cuda_jl','ka','jacc']:
 bp=BASE/m/'plan.json';base=json.loads(bp.read_text());rp=P(base['reference_result']);cp=rp.with_name('config.json');cfg=json.loads(cp.read_text());mc=cfg['models'][m]
 if cfg['gpu']!='5,6'or cfg['input_contract']['ranks']!=2 or cfg['input_contract']['physical_devices']!=[5,6]:raise ValueError('Exactaccepted2devicecontractrequired')
 if base['configuration_id']!=cfg['configuration_id']or sha(rp)!=base['reference_result_sha256']or sha(cp)!=base['raw_config_sha256']:raise ValueError('Selectedresult/configdrift')
 for p,h in base['execution_bindings'].items():
  if sha(p)!=h:raise ValueError('Canonicalbindingdrift '+p)
 bound=dict(base['execution_bindings']);bound[str(rp)]=sha(rp);bound[str(cp)]=sha(cp);bound[str(bp)]=sha(bp);bound[str(envreview)]=sha(envreview)
 for p in [*libs,ncu,nvcc,smi,python,guard,H/'ompi/bin/mpirun',O/'profile_window.h',O/'profile_window.cu',O/'rank_dispatch.py',O/'rank_worker.py',O/'session_entry.py',O/'check_artifact.py']:
  bound[str(p)]=sha(p)
 dest=O/'prepared'/m;dest.mkdir();build=copy.deepcopy(base['build_command']);old=base['scratch_cwd'];build=[x.replace(old,'{scratch}')for x in build];build[0]=str(P(shutil.which(build[0])).resolve())
 if m=='cuda':
  src=SRC/'study_native.cu';orig=src.read_text();v=orig;ch=[]
  def patch(a,b):
   global v
   if v.count(a)!=1:raise ValueError('Exactnativeoccurrence '+a)
   v=v.replace(a,b,1);ch.append({'before':a,'after':b})
  patch('#include <cuda_runtime.h>','#include <cuda_runtime.h>\n#include '+json.dumps(str(O/'profile_window.h')))
  patch('#include "study_support.h"','#include '+json.dumps(str(SRC/'study_support.h')))
  patch('#include "../allreduce-cuda/collectives.cu"','#include '+json.dumps(str(W/'src/allreduce-cuda/collectives.cu')))
  call='kernel_add<<<32,256,0,stream>>>(tmp,out+starts[recv],lens[recv],out+starts[recv]);CK(cudaGetLastError());CK(cudaDeviceSynchronize());'
  patch(call,'profile_before_add(c.n,rank,starts[recv],lens[recv]);'+call+'profile_after_add();')
  patch('for(int trial=1;trial<=5;trial++){study_mpi_guard();','for(int trial=1;trial<=5;trial++){profile_set_trial(trial);study_mpi_guard();')
  patch('study_mpi_finish();return 0;','profile_finish(rank);study_mpi_finish();return 0;')
  inv=v
  for q in reversed(ch):inv=inv.replace(q['after'],q['before'],1)
  if inv!=orig:raise ValueError('Nativeinverse')
  target=dest/'study_native.cu';target.write_text(v);build=[str(target)if x==str(src)else x for x in build];rankargv=['{scratch}/study_native'];proofs[m]={'original':bind(src),'copy':bind(target),'inverse_exact':True,'patches':ch,'native_kernel_source_unchanged':bind(W/'src/allreduce-cuda/collectives.cu'),'native_MPI_protocol_source_unchanged':bind(SRC/'study_mpi_shim.cpp')};bound[str(target)]=sha(target);extra=[];regex=r'^(?:void )?kernel_add\('
 else:
  suffix='cuda.jl'if m=='cuda_jl'else'KA'if m=='ka'else'JACC';src=SRC/f'allreduce-julia-{suffix}/main.jl';orig=src.read_text();before='include(joinpath(@__DIR__,"..","study_common.jl"))';after='include('+json.dumps(str(commoncopy))+')'
  if orig.count(before)!=1:raise ValueError('ExactJuliainclude')
  v=orig.replace(before,after,1);target=dest/'main.jl';target.write_text(v)
  if v.replace(after,before,1)!=orig:raise ValueError('Juliainverse')
  prefix=orig.split(before)[0]
  if v.split(after)[0]!=prefix:raise ValueError('Kernel/dispatchprefixbyteschanged')
  rankargv=['/home/imo/.juliaup/bin/julia',f'--project={W}/src/_julia_jacc_env'if m=='jacc'else f'--project={W}/src/_julia_ka_env',str(target)]
  bound[str(src)]=sha(src);bound[str(target)]=sha(target);bound[str(commoncopy)]=sha(commoncopy);bound[str(P(rankargv[0]).resolve())]=sha(P(rankargv[0]).resolve())
  # Bind prospective installed Julia binary too; retain accepted juliaup argv.
  real=P('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia');bound[str(real)]=sha(real)
  extra=[[str(nvcc),'-O3','-std=c++17','-arch=sm_70','--shared','-Xcompiler=-fPIC',str(O/'profile_window.cu'),'-o','{scratch}/profile_window.so']]
  proofs[m]={'original':bind(src),'copy':bind(target),'inverse_exact':True,'only_entry_change':'include literal redirects to instrumented host common controller','kernel_and_reduce_allocation_prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'prefix_byte_identical':True,'common_host_proof':commonproof,'MPI_protocol_source_unchanged':bind(SRC/'study_mpi_shim.cpp')};regex=r'^add_kernel_\('if m=='cuda_jl'else r'^gpu_add_kernel_\('if m=='ka'else r'^_parallel_for_cuda\([^\n]*\badd_kernel_(?=[,<(])'
 originalcmd=mc['command'];env={}
 if originalcmd[0]=='env':
  for x in originalcmd[1:]:
   if '='not in x:break
   k,z=x.split('=',1);env[k]=z
 env.update({'CUDA_VISIBLE_DEVICES':'5,6','STUDY_OUTPUT_DIR':'{scratch}/profile_outputs','TMPDIR':'{scratch}/tmp','CUDA_CACHE_PATH':'{scratch}/cuda_cache','XDG_CACHE_HOME':'{scratch}/xdg_cache','JULIA_DEPOT_PATH':'{scratch}/julia_depot:/home/imo/.julia','PYTHONDONTWRITEBYTECODE':'1','OMPI_MCA_orte_tmpdir_base':'{scratch}/mpi_tmp','UCX_POSIX_DIR':'{scratch}/ucx_posix'})
 for p in [P(build[0]),SRC/'study_mpi_launch.py']:bound[str(p)]=sha(p)
 args=['--target-processes','application-only','--devices','1','--profile-from-start','off','--filter-mode','global','--kernel-name-base','demangled','--kernel-name','regex:'+regex,'--launch-count','1','--kill','0','--check-exit-code','1','--clock-control','none','--cache-control','all','--replay-mode','kernel','--metrics',','.join(metrics),'--csv','--page','raw','--print-units','base','--log-file','{slot}/counters.csv']
 models[m]={'model':m,'configuration_id':base['configuration_id'],'selected_result':str(rp),'reference_result_sha256':sha(rp),'selected_config':str(cp),'raw_config_sha256':sha(cp),'accepted_model_contract':mc,'execution_bindings':bound,'build_commands':[build,*extra],'rank_target_argv':rankargv,'rank_dispatch':str(O/'rank_dispatch.py'),'rank_worker':str(O/'rank_worker.py'),'session_entry':str(O/'session_entry.py'),'accepted_MPI_launcher':str(SRC/'study_mpi_launch.py'),'environment_overrides':env,'ncu_executable':str(ncu),'ncu_arguments':args,'kernel_regex':regex,'runtime_asset_bytes':0,'payload_bytes_per_trial':21078412,'payload_all5_bytes':105392060,'auxiliary_budget_bytes':536870912,'reserve_bytes':2147483648,'observer_deadline_seconds':1800,'build_deadline_seconds':300,'MPI_registry_startup_deadline_seconds':120,'required_successful_trials':[1,2,3,4,5],'authorized_to_execute':False}
 allbindings.update(bound)
guide=P('/usr/share/doc/nsight-compute/ProfilingGuide/index.html');helpfile=P('/tmp/roofline-completion-20261021/multidevice_roofline_method_review/installed_ncu2022_help.txt');allbindings[str(guide)]=sha(guide);allbindings[str(helpfile)]=sha(helpfile)
for p in [W/'tools/v100_roofline_collect.py',O/'prepare_sources.py',O/'run_proposal.py']:allbindings[str(p)]=sha(p)
dump(O/'source_equivalence.json',proofs);allbindings[str(O/'source_equivalence.json')]=sha(O/'source_equivalence.json')
bundle={'schema_version':1,'status':'READY_FOR_ROOT_REVIEW_NOT_EXECUTION_AUTHORIZED','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'benchmark':'allreduce','physical_GPU_UUIDs':physical,'UUID_metadata_source':bind(envreview),'input_contract':cfg['input_contract'],'models':models,'all_readonly_bindings':allbindings,'source_equivalence_proof':bind(O/'source_equivalence.json'),'write_guard':bind(guard),'runner':bind(O/'run_proposal.py'),'installed_NCU_guide':bind(guide),'installed_NCU_help':bind(helpfile),'common_lock':'/tmp/hecbench-julia-study-gpu.lock','pair_lock':'/tmp/hecbench-allreduce-gpu5-6.lock','permission':'Preparation only. GPU5 roofline execution requires additional explicit user authorization after root independent code review. No authorization inferred from accepted historical2GPUcontract.','runtime_gates':['Both physical5/6 UUID identity+idle before CUDA; continuously monitor both against exactrank/profiler PID+uid+starttime+samefreshsession+executable registry.','OpenMPI/NCU/juliaup must preserve the exact registered worker PID and fresh session across exec; otherwise refuse/fail before accepting counters, no broad process whitelist.','NCU --devices1 under CVD5,6 mapping is not statically asserted: exact runtimecurrentdevice+PCI6 guard and everycounterPCI6+rank1PID mandatory, no fallback or otherdevice data.','Fresh-only Landlock paths; private MPI tmp/UCX backing/cache paths are path controls only. If retained UCX_TLS sm writes elsewhere or newpath option incompatible, fail without unrestricted fallback or transport change.','Singlemarkedrank1localadd after completedrecv+fence beforegather, trial2 n1048576 firstrepeat only. NCU must restore modified privateoutput for eachkernelreplaypass.','Exactly5 original correcttrialrows plus completeindependent bitwise bothrank artifactchecks and bothrank96call/finalmarker guards. Any missing stage/zero/wrongGPUcounter remains failure.'],'limits':'No ccl/NCCL replay inference, no singleGPUrewrite, no primary timing replacement, no completecollective/ROIaggregate or rank/case extrapolation.','GPU_runs':0,'build_runs':0,'tests':0}
dump(O/'proposal.json',bundle);print(json.dumps({'proposal':str(O/'proposal.json'),'sha256':sha(O/'proposal.json'),'models':list(models),'source_proof_sha256':sha(O/'source_equivalence.json'),'GPU_runs':0,'build_runs':0,'tests':0}))
