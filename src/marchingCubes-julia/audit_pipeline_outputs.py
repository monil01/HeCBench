#!/usr/bin/env python3
"""Audit saved artifacts without executing/timing GPU code; standard library only."""
from pathlib import Path
import array,hashlib,json,math,re,struct,sys
P=Path(__file__).resolve().parent;R=P.parent.parent
run=Path(sys.argv[1]);run=run if run.is_absolute()else R/run;r=json.loads(run.read_text())
source=(P.parent/'marchingCubes-cuda/tables.h').read_text()
def section(name):
 s=source[source.index(name):];s=s[s.index('{')+1:s.index('};')];return [int(v,0)for v in re.findall(r'-?0x[0-9a-fA-F]+|-?\d+',s)]
d=section('distinctEdgesTable');t=section('triTable');e=section('edgeIDTable');iso=struct.unpack('<f',struct.pack('<f',-.9))[0]
assert (P/'study_tables_u16.bin').read_bytes()==struct.pack('<256H',*d)
assert (P/'study_tables_i32.bin').read_bytes()==struct.pack('<4096i',*t)
assert (P/'study_edges_u8.bin').read_bytes()==bytes(e)
prefix=struct.pack('<4401d',iso,*d,*t,*e);assert (P/'study_input_prefix_f64.bin').read_bytes()==prefix
expected=array.array('d');expected.frombytes((P/'study_expected_f64.bin').read_bytes());assert sys.byteorder=='little' and len(expected)==401053
out={'run':str(run),'actual_input_binding':'Native copies unchanged header tables and literal Float32 iso. Julia copies exact table files; these files and serialized prefix are byte-checked against parsed original header/literal. No input tolerance applies.','input_prefix_count':4401,'models':{}}
for model,x in r['models'].items():
 rows=[]
 for q in x['trials']:
  f=Path(q['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:35208]==prefix;got=array.array('d');got.frombytes(raw);assert len(got)==401053 and all(map(math.isfinite,got))
  assert got[152369:401049]==expected[152369:401049] # both ID arrays and all4 counts exactly
  mm=max(abs(got[i]-expected[i])for i in range(4401,152369));scores=max(abs(got[i]-expected[i])for i in range(401049,401053));assert mm<=1e-3 and scores<=1e-3
  rows.append({'trial':q['trial'],'prefix_exact':True,'all_active_ids_and_counts_exact':True,'lv1_minmax_max_abs_error':mm,'normalized_partial_coordinate_max_abs_error':scores,'sha256':hashlib.sha256(raw).hexdigest(),'source_artifact':str(f)})
 out['models'][model]=rows
output=R/'.porting-state/porting_logs/marchingCubes-julia-jacc/pipeline_checks.json';output.write_text(json.dumps(out,indent=2)+'\n');print('PASS: all20 exact input prefixes, full active IDs/counts, finite Lv1 minmax and normalized coordinate properties; raw artifacts retained.')
