"""CPU-only source/binary/reference inventory; never imports or runs controller."""
import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');OWN=Path(__file__).resolve().parent;NCUROOT=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Compute');COLLECTOR=ROOT/'tools/v100_roofline_collect.py';REQUIRED='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(COLLECTOR)!=REQUIRED:raise RuntimeError('Rootcollector drift before prospective binding')
controller=OWN/'calibrate_controller.py';nvidia=Path(shutil.which('nvidia-smi')).resolve();sources=[Path('/usr/bin/python3').resolve(),ROOT/'tools/v100_roofline_classify.py',ROOT/'tools/v100_profile_write_guard.py',COLLECTOR,controller,nvidia,OWN/'prepare_manifest.py',NCUROOT/'ncu',ROOT/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json']
sources.extend(p for p in (NCUROOT/'target/linux-desktop-glibc_2_11_3-x64').rglob('*')if p.is_file());sources.extend(p for p in (NCUROOT/'sections').rglob('*')if p.is_file())
cases={};oldcompilers={}
for label,folder,src in [('precision','precision_calibration','precision_canary.cu'),('half_arithmetic','half_arithmetic_calibration','precision_arithmetic_canary.cu')]:
 d=RAW/folder;pr=json.loads((d/'provenance.json').read_text());source=RAW/src
 if sha(source)!=pr['source_sha256']or pr['returncode']!=0:raise RuntimeError('Oldsource/provenance reference mismatch')
 lines=[x for x in (d/'execution.log').read_text().splitlines()if x.startswith('calibration_correct=')]
 if len(lines)!=1 or not lines[0].startswith('calibration_correct=true '):raise RuntimeError('Oldfullhostcheck absent')
 cases[label]={'binary':str(d/'canary'),'source':str(source),'old_counters':str(d/'counters.csv'),'old_execution_log':str(d/'execution.log'),'old_provenance':str(d/'provenance.json'),'expected_host_check_line':lines[0]}
 sources.extend([d/'canary',source,d/'counters.csv',d/'execution.log',d/'provenance.json',d/'build.log']);oldcompilers[label]={'recorded_build_argv':pr.get('build'),'actual_original_compiler_executable_version_or_SHA':None,'no_rebuild':True,'oldprofiler_reference':'2022.4.1.0; exact savedraw counters/hostcheck reused, notnewtiming'}
bindings={str(p.resolve()):sha(p)for p in sources}
node=next(n for n in ast.parse(COLLECTOR.read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='METRICS'for t in n.targets))
manifest={'schema_version':1,'source_root':str(ROOT),'raw_root':str(RAW),'collector_sha256':REQUIRED,'ncu_executable':str(NCUROOT/'ncu'),'ncu_target_ELF':str(NCUROOT/'target/linux-desktop-glibc_2_11_3-x64/ncu'),'nvidia_smi_executable':str(nvidia),'bindings':bindings,'cases':cases,'original_compiler_identity':oldcompilers,'pinned_runtime_environment':{'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'','LD_PRELOAD':''},'pinned_inherited_environment':{},'metrics_source_expression':ast.get_source_segment(COLLECTOR.read_text(),node.value),'full_metrics_required':'All exact collector.METRICS, no unsupported metric dropping','physical_half_lane_convention':{'lane_factor':2,'fma_FLOPs':4,'scalar_source_useful_fma_FLOPs':2,'no_conversion_change':True},'tensor_convention':'VoltaHMMA8842048instructions×512FLOPs=1048576 only','prepared_only':True,'GPU_build_tests_executed':False,'root_adoption_still_required':True}
p=OWN/'source_binding_manifest.json';p.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
argv=['/usr/bin/python3',str(controller),'--source-root',str(ROOT),'--raw-root',str(RAW),'--output-root',str(RAW/'method_calibration_ncu2025_2_0_003'),'--manifest',str(p),'--manifest-sha256',sha(p),'--timeout','600','--reserve-mib','2048','--execute']
d={'controller':str(controller),'controller_sha256':sha(controller),'manifest':str(p),'manifest_sha256':sha(p),'source_bindings':len(bindings),'root_launch_argv':argv,'not_executed':True,'old_compiler_unknown_not_retrocertified':True,'guarded_tool_identity_and_metrics_source':REQUIRED,'no_global_tools_ledgers_primary_measurement_or_method_adoption_changes':True};(OWN/'delivery.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
