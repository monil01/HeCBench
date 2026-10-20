#!/usr/bin/env python3
"""Recreate the named full source pipeline configuration from pinned sources."""
import argparse,json,pathlib,copy
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--reuse-native-result',type=pathlib.Path);p.add_argument('--models',choices=['cuda_jl','ka','jacc','all'],default='all');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent
config=json.loads((B/'native_configuration_template.json').read_text().replace('{benchmark_parent}',str(B)))
config['configuration_id']='fullResyn2_K3_fourSmallAIGs_all15passes_JuliaExplicitKernels_nativeThrustHybrid_v1'
config['implementation_scope_qualification']='EveryexplicitselectedsourcekernelismappedtoauthoredJulia; originalC++CPUcontrollerandnativeThrusthostlibrarysequence/scan/sort/copy/reduceoperationsincludingpredicatesretained. Qualifiedhybrid, notpureframeworkorbackend-portableimplementation.'
# Qualification is implementation metadata; frozen numericalinputcontract unchanged.
main=B/'logic-rewrite-julia-cuda.jl/main.jl';sources=['aig_manager.cu','balance.cu','print.cu','refactor_core.cu','refactor.cu','refactor_mffc.cu','rewrite.cu','strash.cu','truth.cu']
sourcefiles=[str(x)for x in sorted((B/'configured_full').rglob('*'))if x.is_file()]+[str(B/'launch_bridge.h'),str(B/'full_study_ffi.cu'),str(B/'native_study.cu'),str(B/'independent_truth_gold.h')]+[str(x)for x in sorted((B/'logic-rewrite-julia-cuda.jl').glob('*.jl'))]+config['models']['cuda']['source_dependencies']
sourcefiles=list(dict.fromkeys(sourcefiles))
config['models']['cuda_jl']=dict(cwd=str(B/'logic-rewrite-julia-cuda.jl'),build=['nvcc','-std=c++17','-O3','--use_fast_math','-arch={arch}','-shared','-Xcompiler','-fPIC',str(B/'full_study_ffi.cu')]+[str(B/'configured_full'/x)for x in sources]+['-o','study_gpu.so','-lpthread'],command=['julia','--project='+str(B.parent/'_julia_ka_env'),str(main),'study_gpu.so',str(B)],structured=True,bind_all_declared_dependencies=True,source_dependencies=sourcefiles,verification_method='FullSource15Passes_AllActualGraphInputsOutputs_ExactExhaustiveBooleanAndNativeComparison',native_check_description='OriginalindependentfullgraphBooleanoracleonall60stages, allsourceexplicitkernelsgenuineJulia. OriginalC++CPUcontrollers+nativeThrusthostoperationsqualifiedhybrid; noexplicitkernelCUDAnativefallback.')
# Target dispatchers share frozen reviewed CUDA.jl mathematical bodies; each uses
# its own real framework entry/launcher. Add only requested actual target models.
for model,child,env in [('ka','logic-rewrite-julia-KA','_julia_ka_env'),('jacc','logic-rewrite-julia-JACC','_julia_jacc_env')]:
 if a.models not in ('all',model):continue
 m=copy.deepcopy(config['models']['cuda_jl']);d=B/child;m['cwd']=str(d)
 m['command']=['julia','--project='+str(B.parent/env),str(d/'main.jl'),'study_gpu.so',str(B)]
 m['source_dependencies'] += [str(d/'main.jl')]
 m['source_dependencies'] += [str(B.parent/env/x) for x in ['Project.toml','Manifest.toml','LocalPreferences.toml']]
 m['source_dependencies']=list(dict.fromkeys(m['source_dependencies']))
 m['native_check_description']='Full169762fields/prefix5659 andindependentexhaustiveBoolean; oneactualframeworkentryspecializedon50sharedmathbodies/51typedlabels, originalC++CPU/nativeThrusthostlibraryqualifiedhybrid.'
 config['models'][model]=m
config['models']['cuda_jl']['source_dependencies'] += [str(B.parent/'_julia_ka_env'/x)for x in ['Project.toml','Manifest.toml','LocalPreferences.toml']]
if a.reuse_native_result:
 config['reuse_result']=str(a.reuse_native_result.resolve());config['reuse_models']=['cuda'];config['allow_cross_configuration_reuse']=True;config['cross_configuration_reuse_reason']='ImplementationclasslabeladdsqualifiedJuliaexplicitkernels/nativeThrusthybrid; unchangednativecommand,allsources,actualinputcontract,precision,initialprefix5659,timingregionandfive-trialprotocol. Archivednativecaptures/timesretainedwithoutnewexecution.'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(config,indent=2)+'\n');print(a.output)
