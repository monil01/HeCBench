"""Independent high-accuracy-digamma variational update oracle, explicit source phase policy."""
from pathlib import Path
import math,array,struct,json,hashlib
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
EPS=f32(1e-6)

def psi(x):
    if x<=0:raise ValueError('Positive Dirichlet parameters required')
    reciprocal=[]
    while x<32:reciprocal.append(1/x);x+=1
    z=1/(x*x)
    tail=-z/12+z*z/120-z**3/252+z**4/240-z**5/132+691*z**6/32760
    return math.log(x)-1/(2*x)+tail-math.fsum(reciprocal)

assert abs(psi(1)+0.5772156649015329)<1e-13
assert abs(psi(.5)+0.5772156649015329+2*math.log(2))<1e-13
for x in [.5,1,1.25,3,7,16,32,65]:assert abs(psi(x+1)-psi(x)-1/x)<1e-13
fixture=array.array('d');oracle=array.array('d');notes=[]
case_specs=[(7,[0,9,2,5,1],2,1),(33,[1,34,0,3,5],2,3),(65,[0,2,66,5,1,3],3,4)]
for ci,(K,lengths,blocks,iters) in enumerate(case_specs):
    D=len(lengths);indptr=[0]
    for n in lengths:indptr.append(indptr[-1]+n)
    NW=indptr[-1]+1;R=2;alpha=[f32(.5+(t%7)/8) for t in range(K)];weights=[[1+(3*w+5*t)%11 for t in range(K)]for w in range(NW)];weights[0][0]=2**-25;normalizers=[math.fsum(weights[w][t]for w in range(NW))for t in range(K)];beta=[[f32(weights[w][t]/normalizers[t])for t in range(K)]for w in range(NW)];counts=[f32(.5 if ci==0 else .5+.25*(w%3))for w in range(NW)];cols=list(range(NW));train_mask=[0]*NW;val_mask=[255]*NW
    fixture.extend([K,NW,D,blocks,iters,R]);fixture.extend(alpha)
    for row in beta:fixture.extend(row)
    fixture.extend(counts);fixture.extend(cols);fixture.extend(indptr);fixture.extend(train_mask);fixture.extend(val_mask)
    gammas=[[0.]*K for _ in range(D)];grad=[[0.]*K for _ in range(blocks)];newbeta=[[0.]*K for _ in range(NW)];trainloss=[0.]*blocks;valloss=[0.]*blocks;snapshots=[];clamps=[0,0]
    for validation in [False,True]:
        for call in range(R):
            initialize=(R==1) if validation else call==0
            for block in range(blocks):
                for doc in range(block,D,blocks):
                    beg,end=indptr[doc:doc+2]
                    if initialize:gammas[doc]=[a+(end-beg)//K for a in alpha]
                    val_phi_sum=[0.]*K
                    for iteration in range(iters):
                        ng=[0.]*K
                        for record in range(beg,end):
                            w=cols[record];count=counts[record]
                            if not validation or iteration+1==iters:
                                weights_phi=[beta[w][t]*math.exp(psi(gammas[doc][t]))for t in range(K)];den=math.fsum(weights_phi);phi=[v/den for v in weights_phi]
                                for t in range(K):
                                    if validation:val_phi_sum[t]+=phi[t]*count
                                    else:ng[t]+=phi[t]*count
                            if iteration+1==iters:
                                if not validation:
                                    for t in range(K):newbeta[w][t]+=phi[t]*count
                                loss=math.fsum(phi[t]*(math.log(max(beta[w][t],EPS))-math.log(max(phi[t],EPS)))for t in range(K))*count
                                clamps[int(validation)]+=sum(beta[w][t]<EPS or phi[t]<EPS for t in range(K))
                                if validation:valloss[block]+=loss
                                else:trainloss[block]+=loss
                        gammas[doc]=[ng[t]+alpha[t]for t in range(K)]
                    total=math.fsum(gammas[doc]);elog=[psi(v)-psi(total)for v in gammas[doc]]
                    for t in range(K):grad[block][t]+=elog[t]
                    trainloss[block]+=math.fsum(ng[t]*elog[t]for t in range(K));valloss[block]+=math.fsum(val_phi_sum[t]*elog[t]for t in range(K))
        snapshot=[]
        for table in [gammas,grad,newbeta]:
            for row in table:snapshot.extend(row)
        snapshot.extend(trainloss);snapshot.extend(valloss);oracle.extend(snapshot);snapshots.append(snapshot)
        for doc in range(D):
            target=math.fsum(alpha)+(0 if validation else math.fsum(counts[indptr[doc]:indptr[doc+1]]));assert abs(math.fsum(gammas[doc])-target)<1e-10
        if validation:assert all(gammas[doc]==alpha for doc in range(D))
    assert newbeta[-1]==[0.]*K
    beta_mass_err=max(abs(math.fsum(newbeta[w])-R*counts[w])for w in range(NW-1));assert beta_mass_err<1e-10
    notes.append({'topics':K,'words':NW,'docs':D,'blocks':blocks,'iterations':iters,'repeats':R,'document_lengths':lengths,'integer_init_quotients':[v//K for v in lengths],'source_unique_cols_uncontended_locks':True,'unused_terminal_word':NW-1,'epsilon_clamp_counts_train_validation':clamps,'beta_mass_oracle_error':beta_mass_err,'source_validation_final_gamma_equals_alpha':True,'phase_output_counts':[len(x)for x in snapshots],'topic_word_probabilities_normalized_beforeFloat32_rounding':True})
(p/'fixtures_f64.bin').write_bytes(fixture.tobytes());(p/'oracle_f64.bin').write_bytes(oracle.tobytes());(p/'independent_oracle.json').write_text(json.dumps({'method':__doc__,'atol':2e-5,'rtol':1e-4,'input_prefix_count':len(fixture),'numeric_phase_output_values':len(oracle),'digamma_method':'Recurrence to32 then Bernoulli expansion throughx^-12 inFloat64; psi1/psihalf/recurrence analyticcontrols to1e-13','source_phase_policy':'Integer(end-beg)//topics initialization; trainfirstinittruethenfalse,validationretainslasttraininitfalse; allvalidationmask255, gammaresetsalpha. Originaluniqueglobalcols yieldsuncontendedlocks.','loss_scope':'Onlysourcephi(logbeta-logphi)+expectedlogtheta contributions, notcompleteELBOEq15/heldoutlikelihood; nofitquality/convergenceclaim','cases':notes,'primary_reference':'https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf','assets':{'fixture_sha256':hashlib.sha256(fixture.tobytes()).hexdigest(),'oracle_sha256':hashlib.sha256(oracle.tobytes()).hexdigest()}},indent=2)+'\n');print({'prefix':len(fixture),'numericoutputs':len(oracle),'cases':notes})
