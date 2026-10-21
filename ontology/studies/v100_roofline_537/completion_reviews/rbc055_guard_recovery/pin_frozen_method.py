"""CPU-only method dependency freezing/binding; no candidate import or execution."""
from pathlib import Path
import json,hashlib,shutil,os
P=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');OLD=RAW/'batch050-rbc-retained-samples';NEW=RAW/'batch055-rbc-retained-guard-retry';B='rayleighBenardConvection';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(p):p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
frozen=P/'frozen_collect_32a9.py';shutil.copyfile(OLD/B/'cuda/collector_source.py',frozen);frozen.chmod(0o444)
if sha(frozen)!='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c':raise ValueError('Frozen sourceSHA')
# Entire previously reviewed old2022 parser/calibration map; no imports.
calmap=json.loads(Path('/tmp/roofline-completion-20261021/allreduce_capture_adapter/prospective_parser_calibration_bindings.json').read_text());pins={str(frozen):sha(frozen)}
for path,digest in calmap.items():
 if path.endswith('v100_roofline_collect.py'):continue
 if sha(path)!=digest:raise ValueError('Pinned oldmethod dependency changed '+path)
 pins[path]=digest
wrapper=Path('/usr/lib/nsight-compute/ncu');elf=wrapper.parent/'target/linux-desktop-glibc_2_11_3-x64/ncu'
for p in(wrapper,elf):pins[str(p)]=sha(p)
oldprov=json.loads((OLD/B/'cuda/profile_provenance.json').read_text())
if oldprov['profiler_executable']!=str(wrapper)or sha(wrapper)!=oldprov['profiler_executable_sha256']or '2022.4.1.0'not in oldprov['profiler_version']:raise ValueError('ObservedoldNCUidentitymismatch')
pinreview={'schema_version':1,'collector':meta(frozen),'frozen_actual_executable_source':True,'no_code_changes_to_oldcollector':True,'static_standalone_review':{'imports':'Allimportsatmoduleentry arePythonstdlib; no importsrelativeprivatepacket.','__file__':'Usedonly toreadownCOLLECTOR_SOURCE/SHA. Frozenpath truthfullyrecords32a9source.','root_argument':'collectcanonicalizesexplicitabsolute--source-root; writeguard/parser/calibration/rootledger pathsderivefromthatroot, notfrozenfiledirname.','dynamic_roots':'Parser/calibration/library inputsremainexplicitrootpaths nowreadonlyhashbound inall055plan/reviewrecords. Tensorloadernotused; RBCflagreviewcodeunchanged.','NCU_resolution':'OldsourceexecutesbarenCu via processPATH. Futurefollower requires shutil.which(ncu).resolve exactoldwrapper, wrapper+exactx86_64ELF hashes. No global/childPATHmutation; unknown/differentresolution hold. No ncu/version command executed bypreparation.','source_only':'Standalonebehavior inspected as text; no import/syntax/test/run performed.'},'observed050_NCU':{'path':oldprov['profiler_executable'],'sha256':oldprov['profiler_executable_sha256'],'version':oldprov['profiler_version'],'qualification':'Recorded050observation; currentwrapper/ELFbytehashes prospectivelybound, futurecollectorrecordsactualversion. No2025calibration/adoptioninference.'},'execution_bindings':pins,'policy':'OldNCU2022method only. Failclosed upon dependency/path drift; explicitrootreviewrequired for anynewversion/method. SourceboundRBC/full5checks/3600/partialretainedselector unchanged.','sustained_idle_concern':'Oneidlepreflight doesnotproveexclusive35+minute6000step/fivecheckwindow. ForeignGPUactivitymayrecurandcurrentguardmuststopagain; nofuturecompletionassumption, noforeignsignals/reservationclaimed.'}
dump(P/'frozen_method_review.json',pinreview)
rp=P/'profile_window_review.json';review=json.loads(rp.read_text());plans=[]
for m in('cuda','cuda_jl','ka','jacc'):
 pp=NEW/B/m/'plan.json';plan=json.loads(pp.read_text());rec=review['models'][B+'/'+m]
 if plan['execution_bindings']!=rec['execution_bindings']:raise ValueError('Existingreviewplanmapsmismatch')
 for path,digest in {**pins,str(P/'frozen_method_review.json'):sha(P/'frozen_method_review.json')}.items():plan['execution_bindings'][path]=digest;rec['execution_bindings'][path]=digest
 plans.append((pp,plan))
dump(rp,review)
for pp,plan in plans:plan['profile_window_review']['sha256']=sha(rp);dump(pp,plan)
report=json.loads((P/'recovery_review.json').read_text());report['source_rebase_only_note']='Benchmark/kernel/controller/header/fixture bytes andallmethodfields identicalafter050→055pathnormalization; additionalmetadata-onlyreadonly frozenmethod/parser/calibration/NCUbindings restrictfuturedrift. No profiler flag ormathchange.';report['path_mapping']['review_semantics_equality_after_path_normalization']=False;report['path_mapping']['exception']='Only new frozenmethod readonlydependency bindings added consistently toplan/reviewmaps; originalsource/function/window/timing/deadline values unchanged.';report['new_review']=meta(rp);report['frozen_method']=meta(P/'frozen_method_review.json');report['current_collector_at_preparation']=report.pop('current_collector');report['actual_future_collector']=meta(frozen)
for row in report['new_plans']:row['new_plan']=meta(row['new_plan']['path'])
cmd=report['root_only_future_invocation'];cmd[1]=str(frozen);report['root_only_future_invocation']=cmd;dump(P/'root_invocation.json',cmd);dump(P/'recovery_review.json',report)
dump(NEW/'summary.json',[{'model':row['model'],'plan':row['new_plan']['path'],'sha256':row['new_plan']['sha256'],'review_sha256':sha(rp)}for row in report['new_plans']]);print(json.dumps({'frozen_collector':meta(frozen),'review':meta(rp),'methodpins':len(pins),'NCUwrapper':meta(wrapper),'elf':meta(elf)}))
