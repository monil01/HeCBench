#!/usr/bin/env python3
"""CPU-only host-marker copies for retained-trial2/RBC timestep240.
All6000 timesteps, math, callbacks and5oracles remain; noGPU/build/tests.
"""
import pathlib,json,hashlib,copy,argparse,re,difflib,sys
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');S=W/'src/rayleighBenardConvection-julia';HERE=P(__file__).resolve().parent;BASE=P('/tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512/rayleighBenardConvection')
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def hashtext(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(p,a):P(p).write_text(json.dumps(a,indent=2)+'\n')
def devicefunctions(text):
 out={}
 for match in re.finditer(r'__global__\s+void\s+(\w+)\s*\([^;]*?\)\s*\{',text,re.S):
  start=match.end()-1;depth=1;end=start+1
  while depth:
   if text[end]=='{':depth+=1
   if text[end]=='}':depth-=1
   end+=1
  out[match[1]]=hashtext(text[match.start():end])
 return out
START='''    if(study_profile_trial==2 && c==240) {
      if(study_profile_enters!=0 || study_profile_exits!=0 || study_nu_history.size()!=1) {fprintf(stderr,"RBC profile precondition rejected\\n");abort();}
      for(int profile_op=0;profile_op<7;profile_op++)study_profile_counts[profile_op]=0;
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaProfilerStart());
      study_profile_active=true;++study_profile_enters;
      printf("RBC_PROFILE_WINDOW_BEGIN {\\"trial\\":2,\\"zero_based_step\\":240,\\"completed_steps_before\\":240,\\"nu_samples_before\\":%zu,\\"dt_before\\":%.17g,\\"frames_before\\":%.17g}\\n",study_nu_history.size(),double(dt),double(frames));fflush(stdout);
    }
'''
STOP='''    if(study_profile_trial==2 && c==240) {
      CUDA_CHECK(cudaDeviceSynchronize());study_profile_active=false;
      CUDA_CHECK(cudaProfilerStop());++study_profile_exits;
      const unsigned long long profile_expected[]={3,6,2,11,5,0,1};
      for(int profile_op=0;profile_op<7;profile_op++)if(study_profile_counts[profile_op]!=profile_expected[profile_op]){fprintf(stderr,"RBC profile stage count rejected\\n");abort();}
      if(study_profile_enters!=1 || study_profile_exits!=1 || study_nu_history.size()!=2){fprintf(stderr,"RBC profile Nu/window condition rejected\\n");abort();}
      printf("RBC_PROFILE_WINDOW_END {\\"trial\\":2,\\"zero_based_step\\":240,\\"nu_samples_after\\":%zu,\\"authored_dispatch_counts\\":[%llu,%llu,%llu,%llu,%llu,%llu,%llu],\\"authored_dispatches\\":28,\\"whole_1000step_capture\\":false}\\n",study_nu_history.size(),study_profile_counts[0],study_profile_counts[1],study_profile_counts[2],study_profile_counts[3],study_profile_counts[4],study_profile_counts[5],study_profile_counts[6]);fflush(stdout);
    }
'''
GLOBALS='''#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
static int study_profile_trial=-1,study_profile_enters=0,study_profile_exits=0;
static bool study_profile_active=false;
static unsigned long long study_profile_counts[7]={};'''
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-root',type=P,required=True);ap.add_argument('--review-output',type=P,required=True);a=ap.parse_args();out=a.output_root.resolve();reviewpath=a.review_output.resolve()
 if out.exists()or out==P('/tmp')or not out.is_relative_to('/tmp')or reviewpath.exists()or not reviewpath.is_relative_to('/tmp')or reviewpath.is_relative_to(out):ap.error('Fresh /tmp planroot andseparate reviewrequired')
 history_path=HERE/'RBC_retained_trial2_history.json';history=json.loads(history_path.read_text())
 if len(history)!=4 or any(x['nu_event_zero_based_steps']!=[38,240,879]or x['saved_tstep']!=3 or x['saved_frames']!=x['reconstructed_frames']for x in history):raise ValueError('Bound full savedtrial2 Nu event proof required')
 for x in history:
  if sha(x['artifact'])!=x['sha256']:raise ValueError('Savedretainedtrial2 artifactdrift')
 original=(S/'configured_source.cuh').read_text();changes=[];text=original
 def replace(old,new):
  nonlocal text
  if text.count(old)!=1:raise ValueError('Exact source occurrence required '+old[:80])
  text=text.replace(old,new,1);changes.append({'old':old,'new':new})
 replace('#include <cuda_runtime.h>',GLOBALS)
 pattern=r'\{\s*if\(study_cb\)\s*\{\s*RBCall q=\{([0-6]),'
 matches=list(re.finditer(pattern,text));dispatch_sites=[]
 if len(matches)!=18:raise ValueError('Expected18 authoredhostdispatchsites, got'+str(len(matches)))
 for match in matches:
  old=match[0];op=int(match[1]);new='{if(study_profile_active)++study_profile_counts['+str(op)+']; if(study_cb){RBCall q={'+str(op)+','
  # Thereare repeatedidenticalprefixes (Nu rows); exact multiplicity intentionallyrecorded.
  if text.count(old)<1:raise ValueError('Dispatchprefix missing')
  text=text.replace(old,new,1);changes.append({'old':old,'new':new});dispatch_sites.append({'op':op,'line':original[:original.find(old)].count('\n')+1})
 replace('  for(int c = STARTSTEP; c < ENDSTEP; c++) {','  for(int c = STARTSTEP; c < ENDSTEP; c++) {\n'+START)
 old='''      frames = 0.f;
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());'''
 new='''      frames = 0.f;
    }
'''+STOP+'''  }

  CUDA_CHECK(cudaDeviceSynchronize());'''
 replace(old,new)
 inverse=text
 for c in reversed(changes):
  if c['new']not in inverse:raise ValueError('Reverse patch missing')
  inverse=inverse.replace(c['new'],c['old'],1)
 if inverse!=original:raise ValueError('Hostinstrumentation inverse didnot recoverexactsource')
 olddev=devicefunctions(original);newdev=devicefunctions(text)
 if olddev!=newdev or len(olddev)!=7:raise ValueError('All7originalGPUfunctionbytes mustremainexact')
 full_original=(S/'full_study.cu').read_text();before='  int rc=source_main(2,argv);';after='  study_profile_trial=prepare?-1:trial; int rc=source_main(2,argv);'
 if full_original.count(before)!=1:raise ValueError('Exacttrialcallsourceproofrequired')
 full=full_original.replace(before,after,1);before_end=' return 0;\n}'
 after_end=' if(!prepare && (study_profile_enters!=1 || study_profile_exits!=1)) {fprintf(stderr,"RBC profilewindow mustenterandexitonce\\n");return 8;}\n return 0;\n}'
 if full.count(before_end)!=1:raise ValueError('Exactcontrollerendrequired')
 full=full.replace(before_end,after_end,1)
 out.mkdir();models={};plans=[]
 for m in ['cuda','cuda_jl','ka','jacc']:
  bp=BASE/m/'plan.json';base=json.loads(bp.read_text());slot=out/'rayleighBenardConvection'/m;scratch=slot/'scratch';src=scratch/'profile_source';slot.mkdir(parents=True);scratch.mkdir();src.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir()
  if base['blocking_gates']or base['runtime_asset_bytes']:raise ValueError('Exactpreviouslyreviewed zeroasset RBCplanrequired')
  bound=dict(base['execution_bindings'])
  for p,h in bound.items():
   if sha(p)!=h:raise ValueError('Selectedbindingdrift '+p)
  rp=P(base['reference_result']);cp=rp.with_name('config.json')
  if sha(rp)!=base['reference_result_sha256']or sha(cp)!=base['raw_config_sha256']:raise ValueError('Selectedevidencedrift')
  header=src/'configured_source.cuh';adapter=src/'full_study.cu';header.write_text(text);adapter.write_text(full)
  plan=copy.deepcopy(base);old=base['scratch_cwd']
  def move(v):
   if isinstance(v,str):return v.replace(old,str(scratch))
   if isinstance(v,list):return [move(x)for x in v]
   if isinstance(v,dict):return {k:move(x)for k,x in v.items()}
   return v
  for key in ['scratch_cwd','environment_overrides','profile_target_command','build_command']:plan[key]=move(base[key])
  target=scratch/('study_native'if m=='cuda'else'study_gpu.so')
  compiler=['/usr/bin/nvcc','-std=c++17','-O3','-arch=sm_70',str(adapter),'-lcublas','-o',str(target)]
  if m!='cuda':compiler+=['-DSTUDY_SHARED','--shared','-Xcompiler=-fPIC']
  guard=W/'tools/v100_profile_write_guard.py';plan['build_command']=[sys.executable,str(guard),'--allow-dir',str(scratch),'--',*compiler]
  # Same source callback Juliaentries/project/libraryargv. Headersreadsamecanonical
  # mathCUDAincludes; copiedfullstudyinclude resolves copiedconfigured_source.cuh.
  for p in [S/'configured_source.cuh',S/'full_study.cu',S/'build_program.py',P('/usr/bin/nvcc'),guard,P(__file__).resolve(),history_path,header,adapter,bp,rp,cp]:bound[str(p)]=sha(p)
  for x in history:bound[x['artifact']]=x['sha256']
  key='rayleighBenardConvection/'+m
  record={'benchmark':'rayleighBenardConvection','model':m,'configuration_id':base['configuration_id'],'source_result_sha256':base['reference_result_sha256'],'reference_result_sha256':base['reference_result_sha256'],'source_config_sha256':base['raw_config_sha256'],'raw_config_sha256':base['raw_config_sha256'],'execution_bindings':bound,'source_reviewed':True,'runtime_validated':False,'profile_from_start':False,'capture_all_launches_in_window':True,'kernel_id':None,'filter_mode':'global','kernel_name_base':'mangled','launch_count':None,'replay_mode':'kernel','selected_window':{'retained_trial':2,'zero_based_step':240,'ordinal_step':241,'whole_scientific_timestep':True,'Nu_conditional_required':True,'Nu_samples_before':1,'Nu_samples_after':2,'completed_unprofiled_steps_in_same_trial_before':240,'total_controller_runs':6,'steps_each_run':1000,'total_steps':6000,'checked_trials':5,'warmup_runs':1},'source_expected_authored_dispatch_counts':[3,6,2,11,5,0,1],'source_expected_authored_dispatches':28,'actual_NCU_launch_count':None,'compiled_kernel_identity_validation_observed':False,'authored_stage_map':{'0':'ElemMultOmega:3RK3Gcalls, grid63x32 threads256; n497004','1':'ElemMultT:2eachG=6, grid63x32 threads256; n498000','2':'ElemMultNu:top+bottom=2, grid4x1 threads256,n1000','3':'SubOne:3Dz(f)+8Nu=11,grid4x1 threads256,n1000','4':'AddOne:3Dzz+2Nu=5,grid4x1 threads256,n1000','5':'Unusedacceptedcontroller entry:0; do notinventcoverage','6':'Updatedt:onlyfirstG compute_velocity1,grid1 threads1,n1 (Juliawrapinputs498000) '},'host_runtime_asserts':['markerentry/exactNu historybefore once','completealloriginal28authoredhostdispatches exactvector','Nu historyafter2 branchactualtrue','markerexit once andcontrollerfinalentered/exitedonce'],'original_checks_preserved':'Full1000stepwarmup+5full1000steporacles; globalJuliaRBC_COUNTS6*[3000,6000,6,3024,3006,0,1000] unchanged; fullactualinputs/history/allfields guardqualification unchanged.','original_GPU_function_bytes_sha256':olddev,'copied_GPU_function_bytes_sha256':newdev,'inverse_host_patch_recovers_original_header_exactly':True,'source_changes':changes,'full_controller_host_changes':[{'old':before,'new':after},{'old':before_end,'new':after_end}],'original_header_sha256':hashtext(original),'copied_header_sha256':sha(header),'original_fullstudy_sha256':hashtext(full_original),'copied_fullstudy_sha256':sha(adapter),'header_unified_diff':''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(S/'configured_source.cuh'),tofile=str(header))),'controller_unified_diff':''.join(difflib.unified_diff(full_original.splitlines(True),full.splitlines(True),fromfile=str(S/'full_study.cu'),tofile=str(adapter))),'compiler_argv':compiler,'compiler_recipe_source':str(S/'build_program.py'),'compiler_recipe_sha256':sha(S/'build_program.py'),'scope':'PartialalllaunchesinsideONE retainedscientifictimestep; sourceauthored28 !=actualNCUcount. VendorcuBLASkernels andframeworkextras mustbecensused; no whole1000steproofline orcausalclaim.','limitations':['Actualmarkerinterval/counts+full5checks requiretargetexecution; staticreviewnotmeasurement','Additionaldevicefences/profilercalls/hostcounts areprofileinstrumentation, profiledtimings do notreplace primarytimings','Logicalstatefromsource240priorsteps; cachesflushedperkernel andactualprofilecountercontextmustberecorded','Captureallvendorlaunches canstillbeexpensive; timeout/resourcefailure retained, neverdedup silently','GenericJACCclosure isnotname-onlystageproof; sourceop/controller mapping plusactualwindowdispatches required']}
  models[key]=record;plan['execution_bindings']=dict(bound);plan['staged']=True;plan['profile_window_review']={'path':str(reviewpath),'sha256':None,'entry':key};plan['profile_window_context_proposal']={'profile_from_start':False,'capture_all_launches_in_window':True,'selector_arguments':['--profile-from-start','off'],'remove_kernel_id_and_first_per_config_dedup':True,'runtime_validated':False};plan['commands_executed']=[];plans.append((slot,plan))
 review={'schema_version':1,'models':models,'source_reviewed':True,'runtime_validated':False,'source_history_proof':{'path':str(history_path),'sha256':sha(history_path)},'scope':'RBC retainedtrial2 timestep240 sourcebound hostprofilerwindow. Currentactivebatch003 untouched; no actualexecution.','GPU_runs':0,'build_runs':0,'tests':0}
 dump(reviewpath,review);reviewhash=sha(reviewpath);summary=[]
 for slot,plan in plans:
  plan['profile_window_review']['sha256']=reviewhash;dump(slot/'plan.json',plan);summary.append({'model':plan['model'],'plan':str(slot/'plan.json'),'sha256':sha(slot/'plan.json'),'review_sha256':reviewhash})
 dump(out/'summary.json',summary);print(json.dumps({'staged':len(plans),'review':str(reviewpath),'sha256':reviewhash,'GPU_runs':0,'build_runs':0}))
if __name__=='__main__':main()
