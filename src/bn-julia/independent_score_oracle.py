"""Independent K2 score: combinatorial parent sets, direct contingency histograms,
Float64 log-factorials, and a forward Float32/table-quantization error envelope.
No source enumeration helper or source state counter is called.
"""
from pathlib import Path
import array,itertools,math,json,hashlib
P=Path(__file__).resolve().parent

def load(path):
 a=array.array('d');a.frombytes(Path(path).read_bytes());return a

def generate():
 fixture=load(P/'fixture.bin');cpu=load(P/'score_oracle.bin');offset=1;gold=array.array('d');budgets=array.array('d');reports=[];cpuoff=0;u=2**-24
 for ci in range(int(fixture[0])):
  N,D,states,S=map(int,fixture[offset:offset+4]);offset+=4;data=list(map(int,fixture[offset:offset+N*D]));offset+=N*D;lg=fixture[offset:offset+D+2];offset+=D+2
  true=[math.lgamma(i+1)for i in range(D+2)];delta=[abs(x-y)for x,y in zip(lg,true)];maxerr=0.;maxbudget=0.;checked=0
  for node in range(N):
   parents=[n for n in range(N)if n!=node]
   combinations=[c for size in range(5)for c in itertools.combinations(parents,size)];assert len(combinations)==S
   for par in combinations:
    counts=[[0,0]for _ in range(2**len(par))]
    for t in range(D):
     key=sum(data[t*N+p]<<k for k,p in enumerate(par));counts[key][data[t*N+node]]+=1
    exact=0.;bound=0.;quantizedterms=[]
    for n0,n1 in counts:
     exact+=true[n0]+true[n1]-true[n0+n1+1]
     bound+=delta[n0]+delta[n1]+delta[n0+n1+1]
     # Four source Float32 operations per local state, including zero/prior;
     # then at most2^4 Float32 accumulation steps across state scores.
     bound+=(4*u/(1-4*u))*(abs(lg[n0])+abs(lg[n1])+abs(lg[n0+n1+1]))
     quantizedterms.append(lg[n0]+lg[n1]-lg[n0+n1+1])
    bound+=(len(counts)*u/(1-len(counts)*u))*sum(abs(x)for x in quantizedterms)
    bound+=32*math.ulp(max(1.,abs(exact))) # Float64 transcendental/reference safety margin.
    observed=cpu[cpuoff];cpuoff+=1;e=abs(observed-exact);assert e<=bound,(ci,node,par,e,bound)
    gold.append(exact);budgets.append(bound);maxerr=max(maxerr,e);maxbudget=max(maxbudget,bound);checked+=1
  reports.append({'case':ci,'nodes':N,'rows':D,'local_scores_checked':checked,'source_CPU_vs_independent_math_max_abs':maxerr,'largest_forward_budget':maxbudget,'max_actual_LG_table_quantization_abs':max(delta),'scope':'K2uniformstatepriorlocalmarginalscore; no globalBayesianstructureoptimality claim'})
 assert cpuoff==len(cpu)and offset==len(fixture)
 (P/'math_score_oracle.bin').write_bytes(gold.tobytes());(P/'math_score_budget.bin').write_bytes(budgets.tobytes());out={'all_pass':True,'independent':True,'derivation':'Direct histogram/combinations with Float64logfactorials; perpointtablequantization+gamma4localarithmetic+gamma2^paccumulation envelope, Float32unitroundoff2^-24. Not fittedfromGPUoutputs.','reports':reports,'oracle_sha256':hashlib.sha256(gold.tobytes()).hexdigest(),'budget_sha256':hashlib.sha256(budgets.tobytes()).hexdigest()};(P/'independent_score_witness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':generate()
