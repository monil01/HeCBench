"""Recompute saved full outputs, exact guards and actual input prefixes independently."""
from pathlib import Path
import array,struct,math,json,hashlib
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
result=R/'.porting-state/portable_julia/runs/20261018T012546283227Z_simpleSpmv/result.json';d=json.loads(result.read_text());raw=(P/'fixture.bin').read_bytes();off=0

def ev(p):return {'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(fmt,n=1):
 global off
 out=struct.unpack_from('<'+fmt*n,raw,off);off+=struct.calcsize(fmt)*n;return out
nc,=read('I');prefix=array.array('d');oracles=[];case_records=[];witness=None

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
for z in range(nc):
 n,nz,tpr=read('I',3);mat=read('f',n*n);x=read('f',n);row=read('Q',n+1);col=read('Q',nz);val=read('f',nz)
 for a in [mat,x,row,col,val]:prefix.extend(a)
 y=[math.fsum(mat[r*n+c]*x[c]for c in range(n))for r in range(n)]
 assert all(y[r]==math.fsum(val[k]*x[col[k]]for k in range(row[r],row[r+1]))for r in range(n))
 for bs in [32,64,128,256,512,1024]:
  for method in ['dense','csr','vector_csr']:oracles.append(y);case_records.append({'case':z,'threads_per_row':tpr,'blocksize':bs,'method':method,'rows':n})
 if witness is None:
  for r in range(n):
   partial=[]
   for lane in range(tpr):
    s=0.
    for k in range(row[r]+lane,row[r+1],tpr):s=f32(s+f32(val[k]*x[col[k]]))
    partial.append(s)
   offset=tpr//2
   while offset:
    partial=[f32(partial[i]+partial[i+offset if i+offset<tpr else i])for i in range(tpr)];offset//=2
   if len(set(partial))>1:
    witness={'case':z,'row':r,'threads_per_row':tpr,'all_lane_store_values_FP32_CPU_emulation':partial,'source_all_lanes_write_same_address':True,'independent_row_dot_FP64':y[r],'observed_original_GPU_failure':False,'actual_original_all_five_pass':True,'conclusion':'Source-leveldifferentvaluessameoutputwriteconflict, originalhardwarefivepassdoesnotguaranteeCUDA-definedwinner. Minimal lane0guardrestoresintendeddefinedownership.'};break
assert witness is not None and off==len(raw)
pre=prefix.tobytes();rows=[]
for m,v in d['models'].items():
 assert v['complete']and len(v['trials'])==5
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);rawout=f.read_bytes();assert rawout[:len(pre)]==pre;out=array.array('d');out.frombytes(rawout[len(pre):]);pos=0;err=0.;rel=0.
  for y in oracles:
   for e in y:
    a=out[pos];pos+=1;assert math.isfinite(a)and abs(a-e)<=1e-5+1e-5*abs(e);err=max(err,abs(a-e));rel=max(rel,abs(a-e)/max(abs(e),1e-30))
   assert all(a==16777215. for a in out[pos:pos+16]);pos+=16
  assert pos==len(out)==47520
  rows.append({'model':m,'trial':t['trial'],'artifact':ev(f),'actual_inputs_lossless_prefix_values':1437701,'complete_output_guard_values':47520,'guards_checked_exact':1440,'max_abs_error_vs_independent_FP64':err,'max_relative_error_vs_independent_FP64_floor1e_30':rel})
original=R/'.porting-state/portable_julia/runs/20261018T011959158673Z_simpleSpmv/result.json';old=json.loads(original.read_text());assert old['models']['cuda']['complete']and all(t['correct']for t in old['models']['cuda']['trials'])
report={'result':ev(result),'all20_complete_captures_pass':True,'original_native_diagnostic':ev(original),'original_native_all_five_pass_despite_source_hazard':True,'single_writer_source_witness':witness,'independent_method':'Dense math.fsum overeachrow, independentlyCSRproductcheck; decodedactualFP32fixture; all20actualinputprefixbytesexact, outputtolerance1e-5/1e-5 andexactguards.','guard_checker_gap':{'primary_self_and_cross_comparisons_use_tolerance_for_all_values':True,'CPU_counterexample_guard_plus1_passes_tolerance_but_not_exact_contract':abs(16777216.-16777215.)<=1e-5+1e-5*16777215.,'independent_exact_guard_check_all20_pass':True,'not_observed_GPU_guard_failure':True},'checks':rows,'path_order':case_records};(P/'independent_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n');print({'all20_pass':True,'max_abs_error':max(r['max_abs_error_vs_independent_FP64']for r in rows),'original_native_pass':True,'source_conflict_witness':witness})
