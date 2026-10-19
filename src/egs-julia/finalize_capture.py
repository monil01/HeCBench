"""Independent gates/serialization outside every model's GPU timing regions."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys
from assemble_artifact import assemble
p=argparse.ArgumentParser();p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--trial',type=int,required=True);a=p.parse_args();D=a.capture_dir;H=Path(__file__).resolve().parent;M=json.loads((D/'capture_manifest.json').read_text())
for cap in M['captures']:
 assert cap['guards_pass'] and cap['all_readonly_unchanged']
 for rec in cap['records'].values():
  f=D/rec['path'];assert f.stat().st_size==rec['bytes'];rec['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
(D/'capture_manifest.json').write_text(json.dumps(M,indent=2)+'\n');check=D/'independent_check.json';subprocess.run([sys.executable,str(H/'check_full_physics.py'),'--capture-dir',str(D),'--output',str(check)],stdout=subprocess.DEVNULL,check=True);review=json.loads(check.read_text())
if not Path('mtcheck').exists():subprocess.run(['g++','-O3','-std=c++17',str(H/'check_mt_trajectory.cpp'),'-o','mtcheck'],check=True)
with(D/'independent_mt_check.json').open('w')as f:subprocess.run(['./mtcheck',str(D)]+[cap['label']for cap in M['captures'][1:]],stdout=f,check=True)
artifact=f'output_trial{a.trial}.bin';enc=assemble(D,artifact);row={'trial':a.trial,'time_seconds':sum(M['segment_seconds']),'correct':review['all_pass'],'checked_values':review['complete_pixel_values_independently_CPU_accumulated'],'max_abs_error':review['all_pixels_accumulation_max_absolute_error'],'max_rel_error':None,'timing_region':'sum_source_simulation_and_detector_accumulation_contextcompleted_segments_excludes_captures','jit_excluded':True,'internal_repeats':1,'output_artifact':{'path':artifact,'dtype':'float64','count':enc['count'],'byte_order':'little'},'input_prefix_count':enc['input_prefix_count'],'segment_seconds':M['segment_seconds'],'actual_histories':review['history'][-1]['started_cumulative'],'inner_iterations_by_phase':[x['inner_iterations']for x in review['history']],'independent_squared_direction_norm_max_error':review['max_particle_direction_squared_norm_error'],'metric_scope':'max_abs_error isfullimageCPUaccumulationerror only; noindependentfulltransporttrajectoryerror computed.'};print('STUDY_TRIAL '+json.dumps(row,allow_nan=False),flush=True)
