from pathlib import Path
import numpy as np,json,hashlib
B=Path(__file__).resolve().parent;R=B.parents[1];W=R/'.porting-state/portable_julia/bonds_reference_review'
f=np.memmap(W/'source_inputs_u32.bin',dtype='<u4',mode='r');N=1000000;off=0;parts=[]
for size in [16,16,4,4,1,9,1]:parts.append(f[off:off+N*size].reshape(N,size));off+=N*size
assert off==len(f)
# Default source payments all have day<=28: the last coupon period is exact six-month advance.
mat=parts[5][:,4:8].astype(np.int64);settle=parts[2].astype(np.int64);rate=parts[5][:,8].copy().view('<f4').astype(np.float64);target=parts[4][:,0].copy().view('<f4').astype(np.float64)
assert np.all((mat[:,1]>=1)&(mat[:,1]<=28));assert np.all(settle[:,0]==mat[:,0])and np.all(settle[:,1]==mat[:,1]-1)and np.all(settle[:,2]==mat[:,2]);assert np.all(mat[:,3]-settle[:,3]==1)
m1=mat[:,0]-6;y1=mat[:,2]-(m1<1);m1=np.where(m1<1,m1+12,m1);d1=mat[:,1]
def days(a_d,a_m,a_y,b_d,b_m,b_y):
 b31=(b_d==31)&(a_d<30);d2=np.where(b31,1,b_d);m2=b_m+b31
 return 360*(b_y-a_y)+30*(m2-a_m-1)+np.maximum(0,30-a_d)+np.minimum(30,d2)
period=days(d1,m1,y1,mat[:,1],mat[:,0],mat[:,2])/360.;elapsed=days(d1,m1,y1,settle[:,1],settle[:,0],settle[:,2])/360.;time=days(settle[:,1],settle[:,0],settle[:,2],mat[:,1],mat[:,0],mat[:,2])/360.;assert np.all(time>0)
coupon=100*rate*period;accrued=100*rate*elapsed;dirty=target+accrued;redemption=100+coupon
yield_=2*np.expm1(np.log(redemption/dirty)/(2*time));npv=redemption*np.exp(-2*time*np.log1p(yield_/2));assert np.allclose(npv,dirty,rtol=3e-15,atol=3e-12)
# Compare actual original CPU outputs before GPU/tolerance freeze, separate from independent construction.
source=np.fromfile(W/'source_CPU_results_f32.bin',dtype='<f4').reshape(5,256);ind=[dirty[:256],accrued[:256],target[:256],yield_[:256],yield_[:256]];summary=[]
for i,want in enumerate(ind):
 actual=source[i].astype(np.float64);e=np.abs(actual-want);summary.append(dict(field=i,max_abs=float(e.max()),max_rel=float((e/np.maximum(1e-20,np.abs(want))).max())))
print(json.dumps(dict(N=N,time_range=[float(time.min()),float(time.max())],yield_range=[float(yield_.min()),float(yield_.max())],actual_CPU_first256_error=summary),indent=2))
for name,arr in [('default_oracle_dirty_f64.bin',dirty),('default_oracle_accrued_f64.bin',accrued),('default_oracle_clean_f64.bin',target),('default_oracle_yield_f64.bin',yield_)]:arr.astype('<f8').tofile(W/name)
(W/'independent_default_oracle_review.json').write_text(json.dumps(dict(scope='Independent Float64 closed-form solution of two final same-date cashflows under source30/360anddefaultsemiannualyield conventions; actualdefaultall1Msource dates checked, no sourceCPUoutputs usedtoconstructoracle.',n=N,raw_dayzero_settlements=int((settle[:,1]==0).sum()),first256_source_CPU_errors=summary,engineering_budget_before_GPU='2e-3+1e-4relative numericalartifact; exactinputs/metadata/guards separately; independentprice/NPVresidual<=2e-3. Forwardyieldconditioningreportedseparately, notmarketrealism oruniformfinancialaccuracyclaim.'),indent=2)+'\n')
