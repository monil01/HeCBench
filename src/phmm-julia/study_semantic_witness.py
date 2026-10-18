from pathlib import Path
from fractions import Fraction
import json,hashlib,shutil
R=Path('/home/imo/HeCBench_julia');p=R/'src/phmm-julia';sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
mask={(1,1)};reads=[(0,1),(0,2),(1,1),(1,2)];missing=[x for x in reads if x not in mask];assert len(missing)==3
poison=[]
for v in [0,1]:
 b={(i,j):[Fraction(v),Fraction(v)]for i,j in reads};b[1,1]=[Fraction(1,4)]*2
 out=sum(sum(b[x])*Fraction(1,4)*Fraction(1,4)for x in reads)*Fraction(1,2)
 poison.append({'initialized_B_background':v,'second_call_output_eachstate':str(out),'Float64_exact':float(out)})
w={'source_hashes':{str(q.relative_to(R)):sha(q)for q in [R/'src/phmm-cuda/main.cu',R/'src/phmm-cuda/kernel.h',R/'src/phmm-cuda/constants_types.h']},'static_proof':{'initial_A':'FullydefinedbyCPUfixture','initial_B':'cudaMalloconly,noinitialization','call1':[1,1],'writes_B':[[1,1]],'call2':[1,2],'B_cells_read':reads,'unwritten_B_cells_read':missing,'undefined_components_per_batch':6,'method':'Written-set/data-dependency proof from actualcallerwholebufferswap andnativekernelreadsites, NOTobservedGPUfailure.'},'safe_initialized_poison_prediction':poison,'witness_assumptions':'Aall1,transition1/4,likelihood1/4,emission1/2, batch4/twostates; call1 B[1,1]=1/4 then call2=1/64 ifbackground0 vs13/64 ifbackground1; valuesdyadicexact. Safe initializedtwofixtures, originaluninitializedGPUrunnotneeded.','scientific_scope':'Definedsourcepartial-update/stale-buffer4-neighbor+selfterm recurrence, nonnormalizedinputs. No standardPairHMMprobability/likelihoodqualitycertificate. j0branchnevercalledbyoriginalCLI, separateunmeasuredsourceboundary.'}
(p/'source_initialization_witness.json').write_text(json.dumps(w,indent=2)+'\n');print(poison)
