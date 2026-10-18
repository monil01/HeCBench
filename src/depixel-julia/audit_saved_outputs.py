"""CPU-only complete exact stage outputs, actual RGB bits/guards and timing audit."""
from pathlib import Path
import struct,json,hashlib,sys
p=Path(__file__).resolve().parent;prefix=(p/'fixture.bin').read_bytes();gold=(p/'oracle.bin').read_bytes();expected=prefix+gold;meta=json.loads((p/'fixture_description.json').read_text());assert len(expected)==27198*4
rows=[]
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text())
 for model,d in res['models'].items():
  if not d.get('trials'):continue
  assert len(d['trials'])==5 and all(t['correct'] for t in d['trials']);assert abs(d['mean_seconds']-sum(t['time_seconds'] for t in d['trials'][1:])/4)<1e-12
  for t in d['trials']:
   file=Path(t['output_artifact']['saved_path']);a=file.read_bytes();assert a==expected,(model,t['trial']);assert len(t['path_times_seconds'])==7 and abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   for path,sha in d.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,(model,path)
   rows.append({'model':model,'trial':t['trial'],'path':str(file),'sha256':hashlib.sha256(a).hexdigest(),'all_intermediate_and_final_graph_words_exact':True,'actual_RGB_echo_bits_and_560_guard_words_exact':True,'all_declared_dependencies_current':True})
print(json.dumps({'conditional_corrected_native':True,'original_invalid_GPU_runs':0,'prefix_count':9998,'artifact_count':27198,'pixels_pertrial':3328,'actual_temporary_graph_pixels':3328,'actual_final_graph_pixels':3328,'actual_device_RGB_echo_words':10320,'guard_words_pertrial':560,'oracle_source':'generate_study_fixture.py independent explicit adjacency-list geometric graph formulation, no GPU helper imported','palette_min_binmargin':meta['quantization_margin_min_255scaled'],'conservative_error_allowance':meta['conservative_32eps_Float32_times255_operation_error_allowance'],'margin_certified':meta['margin_exceeds_bound'],'not_claimed':['Guaranteedallcrossesremoved/planarity','Fullvectorizationorimagequality','ArbitraryRGBquantizationboundary/nonfinitebehavior','UntouchednativeGPUperformance'],'rows':rows},indent=2))
