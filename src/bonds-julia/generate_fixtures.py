from pathlib import Path
import numpy as np,json,hashlib,sys
B=Path(__file__).resolve().parent;R=B.parents[1];W=R/'.porting-state/portable_julia/bonds_reference_review';f=np.memmap(W/'source_inputs_u32.bin',dtype='<u4',mode='r');N=1000000;off=0;fields=[]
for size in [16,16,4,4,1,9,1]:fields.append(f[off:off+size*N].reshape(N,size));off+=size*N
ind=[np.memmap(W/name,dtype='<f8',mode='r')for name in ['default_oracle_dirty_f64.bin','default_oracle_accrued_f64.bin','default_oracle_clean_f64.bin','default_oracle_yield_f64.bin','default_oracle_yield_f64.bin']]
cs=[];io=oo=0
with (B/'input_u32.bin').open('wb')as out,(B/'oracle_f64.bin').open('wb')as oracle:
 for n in [1,17,257,N]:
  cs.append(dict(n=n,io=io,oo=oo,source_full_shape=n==N,initializer='FirstNactualsource defaultseed123 rows; each sourceCLIcallreseedssame123',all_future_cashflows_at_single_maturity=True));io+=51*n;oo+=5*n
  for a in fields:np.asarray(a[:n],dtype='<u4').tofile(out)
  for a in ind:np.asarray(a[:n],dtype='<f8').tofile(oracle)
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
assets=[dict(path=str((B/n).relative_to(R)),bytes=(B/n).stat().st_size,sha256=sha(B/n))for n in ['input_u32.bin','oracle_f64.bin']]
(B/'cases.json').write_text(json.dumps(dict(cases=cs,assets=assets,CPU_ABI=json.loads((W/'ABI.json').read_text()),source_generation=json.loads((W/'source_generation.json').read_text()),independent_oracle=json.loads((W/'independent_default_oracle_review.json').read_text()),numpy=np.__version__,python=sys.version,pre_GPU_budget=dict(atol=.002,rtol=.0001,price_and_NPV_abs_budget=.002,all_guards_inputs_readonly_exact=True),limits='Originalsourcecalendar/date0fields preserved; one-day sourceyieldregime extremeandconditioned, no realisticmarketcertification. CPUoracles neveruploaded. Allactualmatrixfields capture; optionalbranch financialconventionsunmeasured.'),indent=2)+'\n')
(B/'study_cases.h').write_text('struct Case {int n,io,oo;};\nconst Case cases[]={'+','.join('{'+','.join(str(c[k])for k in ['n','io','oo'])+'}'for c in cs)+'};\n')
(B/'study_cases.jl').write_text('const CASES=['+','.join('(;'+','.join(k+'='+str(c[k])for k in ['n','io','oo'])+')'for c in cs)+']\n')
print(assets,'prefix',sum(55*c['n']+352 for c in cs))
