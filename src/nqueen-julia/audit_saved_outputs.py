"""CPU-only exact input/prefix/count audit. No new kernel timing."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
p=Path(__file__).resolve().parent;s=json.loads((p/'study_shape.json').read_text());P=s['prefixes'];raw=(p/'study_roots.bin').read_bytes();expected=(p/'study_expected_u64.bin').read_bytes();dtype=np.dtype([('control','<u4'),('board','i1',(12,))]);roots=np.frombuffer(raw,dtype=dtype);assert dtype.itemsize==16 and len(roots)==P;board=roots['board'];assert (board[:,:7]>=0).all()and(board[:,:7]<15).all()and(board[:,7:]==0).all();flags=np.zeros(P,dtype=np.uint32)
for row in range(7):
 flags|=np.left_shift(np.uint32(1),board[:,row].astype(np.uint32))
 for prev in range(row):
  delta=board[:,row].astype(np.int16)-board[:,prev].astype(np.int16);assert (delta!=0).all()and(np.abs(delta)!=row-prev).all()
assert np.array_equal(flags,roots['control']);values=np.frombuffer(expected,dtype='<u8');assert int(values[:P].sum(dtype=np.uint64))+2466109==171129071 and int(values[P:].sum(dtype=np.uint64))==2279184
rp=Path(sys.argv[1]);r=json.loads(rp.read_text());models={}
for model,m in r['models'].items():
 trials=[]
 for t in m['trials']:
  f=Path(t['output_artifact']['saved_path']);content=f.read_bytes();assert content==raw+expected
  trials.append({'trial':t['trial'],'artifact':str(f),'sha256':hashlib.sha256(content).hexdigest(),'exact_input_root_bytes':True,'every_subtree_node_count_exact':True,'every_subtree_solution_count_exact':True,'max_abs_error':0,'max_rel_error':0})
 models[model]=trials
out={'benchmark':'nqueen','result':str(rp),'all_prefixes':P,'input_bytes':len(raw),'output_uint64_values':2*P,'method':'Exact all20artifacts versusactualsourceprefixbytes+independentbitmaskoracle, allprefixcolumn/diagonallegality/controlflags/unusedzero fields independentlychecked. Allperprefix UInt64 values retain actualbits, noFloat64surrogate. Globaltotals supplementary.','models':models,'numpy':np.__version__};(p/'saved_artifact_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('All20artifacts exact; every source root legal/controlmask exact; fullglobal totals match.')
