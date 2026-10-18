"""Source-independent Float64 pair sums and a priori FP32 forward-error model."""
from pathlib import Path
import array,struct,math,json,hashlib
p=Path(__file__).resolve().parent
f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
u=2.0**-24
gamma=lambda n:n*u/(1-n*u)
fixture=array.array('d');oracle=array.array('d');budget=array.array('d');notes=[]
for mode,B in enumerate([1,2,3]):
    boxes=B**3;particles=100*boxes;neighbors=[];metadata=[]
    for bz in range(B):
        for by in range(B):
            for bx in range(B):
                number=(bz*B+by)*B+bx;nei=[]
                for dz in [-1,0,1]:
                    for dy in [-1,0,1]:
                        for dx in [-1,0,1]:
                            x,y,z=bx+dx,by+dy,bz+dz
                            if 0<=x<B and 0<=y<B and 0<=z<B and (dx or dy or dz):nei.append((x,y,z,(z*B+y)*B+x))
                raw=struct.pack('<iiiiqi4x',bx,by,bz,number,100*number,len(nei))
                for x,y,z,n in nei:raw+=struct.pack('<iiiiq',x,y,z,n,100*n)
                raw+=bytes(24*(26-len(nei)));assert len(raw)==656;words=array.array('I');words.frombytes(raw);metadata.extend(words);neighbors.append([number]+[q[3] for q in nei])
    rv=[];charges=[]
    for box in range(boxes):
        bx=box%B;by=(box//B)%B;bz=box//(B*B)
        for part in range(100):
            if mode==0:v=x=y=z=0.;q=(part%16+1)/16
            elif mode==1:
                i=100*box+part;v=f32(((i*7+1)%10+1)/10);x=f32(((i*3+2)%10+1)/10);y=f32(((i*9+4)%10+1)/10);z=f32(((i*11+6)%10+1)/10);q=f32(((i*13+3)%10+1)/10)
            else:
                x=(8*bx+(part%5+1))/64;y=(8*by+((part//5)%5+1))/64;z=(8*bz+((part//25)%4+1))/64;v=(x*x+y*y+z*z)/2;q=(part%8+1)/8
            rv.append(tuple(map(f32,[v,x,y,z])));charges.append(f32(q))
    fixture.extend([B,mode,.5,particles,len(metadata)]);fixture.extend(metadata)
    for r in rv:fixture.extend(r)
    fixture.extend(charges);outputs=[];bounds=[];max_r2=0.;min_r2=math.inf
    for home in range(boxes):
        pair_indices=[100*n+j for n in neighbors[home] for j in range(100)];count=len(pair_indices)
        for particle in range(100):
            i=100*home+particle;va,xa,ya,za=rv[i];terms=[[],[],[],[]];terrors=[[],[],[],[]]
            for j in pair_indices:
                vb,xb,yb,zb=rv[j];q=charges[j];dot=math.fsum([xa*xb,ya*yb,za*zb]);r2=va+vb-dot;min_r2=min(min_r2,r2);max_r2=max(max_r2,r2);phi=math.exp(-.5*r2)
                # Conservative scalar-expression error, valid with or without FMA contraction.
                er=gamma(8)*(abs(va)+abs(vb)+abs(xa*xb)+abs(ya*yb)+abs(za*zb));eu=.5*er+u*abs(.5*r2);ephi=phi*(math.expm1(eu)+16*u*math.exp(eu));potential=q*phi;terms[0].append(potential);terrors[0].append(abs(q)*ephi+gamma(2)*(abs(potential)+abs(q)*ephi))
                for axis,(a,b) in enumerate([(xa,xb),(ya,yb),(za,zb)],1):
                    d=a-b;ed=u*(abs(a)+abs(b));term=2*q*phi*d;et=2*abs(q)*((abs(d)+ed)*ephi+phi*ed);terms[axis].append(term);terrors[axis].append(et+gamma(4)*(abs(term)+et))
            for ts,es in zip(terms,terrors):
                value=math.fsum(ts);eb=math.fsum(es)+gamma(count)*(math.fsum(map(abs,ts))+math.fsum(es))+1e-12*(1+abs(value));outputs.append(value);bounds.append(eb)
    if mode==0:
        total=math.fsum(charges);assert all(outputs[4*i]==total and all(v==0 for v in outputs[4*i+1:4*i+4]) for i in range(particles))
    weighted_net=[math.fsum(charges[i]*outputs[4*i+k] for i in range(particles)) for k in [1,2,3]];assert max(map(abs,weighted_net))<1e-9
    oracle.extend(outputs);budget.extend(bounds);notes.append({'boxes1d':B,'mode':mode,'particles':particles,'metadata_bytes':len(metadata)*4,'neighbor_counts_including_home':sorted({len(n) for n in neighbors}),'positive_charges':True,'oracle_weighted_net_vector_sum':weighted_net,'min_r2':min_r2,'max_r2':max_r2,'max_abs_oracle_output':max(map(abs,outputs)),'max_pointwise_forward_error_budget':max(bounds),'fixture_scope':['collocated_exactsum_zerovector','source_style_independent_v_and_coordinates','v_equals_half_squarednorm_Gaussian_compatible'][mode]})
(p/'fixtures_f64.bin').write_bytes(fixture.tobytes());(p/'oracle_f64.bin').write_bytes(oracle.tobytes());(p/'error_budget_f64.bin').write_bytes(budget.tobytes())
derivation={'method':__doc__,'pre_measurement':True,'unit_roundoff':u,'gamma_n':'n*u/(1-n*u)','r2_error':'gamma8*(|vA|+|vB|+|xAxB|+|yAyB|+|zAzB|)','exp_argument_error':'.5*r2error +u*abs(.5*r2), alpha=.5 and a2=.5 exact','exp_error':'phi*(expm1(argumenterror)+16u*exp(argumenterror)); conservative safety factor above documented standardexpf2ULP plus correct-rounding error, not fast__expf','perterm_error':'Potential weightedexp error plusgamma2 product error; vector displacement subtraction error plusweightedexp error andgamma4 products. See independentgenerator for exact executable formulas.','sum_error':'sum(pertermerror)+gamma_actual_paircount*(sum(abs(true_terms))+sum(pertermerror))+1e-12*(1+abs(Float64fsumoracle))','assumptions':'Normal finite FP32 round-to-nearest operations; no fastmath; documented standard expf accuracy. SourceandtargetmaycontractFMA, which this conservative operation count covers. This conditional engineering forward-error model is not a formal library proof or application-specific MD accuracy certificate.','dual_reference_tolerance':{'atol':0.0002,'rtol':0.00002},'exact_controls':'Collocatedfixtureforcesexactzeroandpotentialexactcharge_sum; allinputs/canariesexact.','physics_scope':'Sourcev independent from coordinates in original; onlythirdfixture imposeshalf-norm². Sourcevector is scaledGaussiannegativegradient (2/a2 factor), not a generalMDforce-law/trajectory certificate.','reference':'https://docs.nvidia.com/cuda/archive/12.2.0/pdf/CUDA_C_Programming_Guide.pdf'}
(p/'error_model_derivation.json').write_text(json.dumps(derivation,indent=2)+'\n');(p/'independent_oracle.json').write_text(json.dumps({'method':__doc__,'input_prefix_count':len(fixture),'output_values':len(oracle),'cases':notes,'pointwise_budget':'error_budget_f64.bin','derivation_sha256':hashlib.sha256((p/'error_model_derivation.json').read_bytes()).hexdigest(),'scope':'Fullconfiguredsourcepairaccumulation, not MDtimeintegration/statisticalmechanicsthermodynamics','assets':{'fixture_sha256':hashlib.sha256(fixture.tobytes()).hexdigest(),'oracle_sha256':hashlib.sha256(oracle.tobytes()).hexdigest(),'budget_sha256':hashlib.sha256(budget.tobytes()).hexdigest()}},indent=2)+'\n');print({'prefix':len(fixture),'outputs':len(oracle),'cases':notes})
