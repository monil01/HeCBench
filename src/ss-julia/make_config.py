#!/usr/bin/env python3
from pathlib import Path
import argparse,json,uuid
P=Path(__file__).resolve().parent;R=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('--models',default='cuda');a.add_argument('--output',required=True);a.add_argument('--reuse-result');a.add_argument('--fixtures',type=Path,default=Path('/tmp/hecbench-stencil-ports-ss-fixtures'));z=a.parse_args();F=z.fixtures.resolve();old=json.load(open(Path(z.reuse_result).resolve().with_name('config.json')))if z.reuse_result else{};capture_root=old.get('capture_root','/tmp/hecbench-stencil-ports-ss-private/campaign_'+uuid.uuid4().hex)
prep=json.load(open(F/'native_preparation_report.json'));assert prep['pass']is True;cases=json.load(open(F/'cases.json'));assets=[F/'cases.json',F/'cases.tsv',F/'canonical_actual_input_prefix.u32.bin',F/'native_preparation_report.json']+[Path(c[k])for c in cases for k in ['text_path','oracle_path']];assets=list(dict.fromkeys(assets))
shared=[P/x for x in ['full_study.cu','configured_main.cuh','study_interface.h','build_program.py','verification_plan.md','reference_provenance.json','source_archive.json','asset_provenance.json','host_compaction_witness.json']]+[x for x in(P/'reference_history').iterdir()if x.is_file()]+assets
c=dict(benchmark='ss',configuration_id='originalOpenCL12_R20000_fullsearch_and6controls_initialized_memmove_sourcefinal_v1',configuration_role='primary',gpu=6,timeout=1200,atol=0,rtol=0,input_prefix_count=prep['input_prefix_count'],raw_output_identity_claim=False,numerical_agreement_kind='complete_source_final_numerical_outputs_with_lossless_raw_bijection',inputs_shared_verified=True,timed_regions_equivalent_verified=True,native_reference_repaired=True,exclude_untouched_native_aggregates=True,reference_provenance=json.load(open(P/'reference_provenance.json')),capture_root=capture_root,models={})
for m in z.models.split(','):
 native=m=='cuda';tag={'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}.get(m);cwd=P if native else P/('ss-julia-'+tag);command=['./study_native',str(F),str(Path(capture_root)/m),'0']if native else['julia','--project='+str(R/'src'/('_julia_jacc_env'if m=='jacc'else'_julia_ka_env')),str(cwd/'main.jl'),str(F),str(Path(capture_root)/m)]
 c['models'][m]=dict(cwd=str(cwd),build=['python3',str(P/'build_program.py'),'--mode','native'if native else'shared','--output','study_native'if native else'study_gpu.so','--arch','{arch}'],command=command,structured=True,bind_all_declared_dependencies=True,source_dependencies=[str(x)for x in shared]+([]if native else[str(P/'study_common.jl'),str(cwd/'main.jl')]),verification_method='CompleteSourceFinalSortedPositions_AllGroupCounts_ExactFullInputReadbacks_RawLosslessBijection_AllGuardsInactive',native_check_description='FullsourceR20000OpenCL1.2doc/22byteAPIpattern +6R1source-modecontrols, sourceCPUreferenceandfinalsortretained; completeactualrawbuffers+inversepermutation retained andlosslesslyreconstructed, independentall-positionASCIIoracle, exactallsourcefinalnumericvalues. Rawatomicorderidentityfalse. Conditionaloutputinitialization/hostmemmoverepair; twoGPUbodyfilesbyteidentical, originalCPU/CABIcontrollerretained.')
 for_target=c['models'][m]
 if not native:
  for_target.pop('build',None)
  for_target['native_check_description']=for_target['native_check_description'].replace('originalCPU/CABIcontrollerretained.','JuliahosttranslationofsourceBMH/launchloops/overlap-safecompactionretained; noCABI/nativeGPUfallback.')
if z.reuse_result:
 c.update(reuse_result=str(Path(z.reuse_result).resolve()),reuse_models=[m for m in c['models']if m!=z.models.split(',')[-1]])
 for m in c['reuse_models']:c['models'][m]=old['models'][m]
p=Path(z.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(c,indent=2)+'\n');print(p)
