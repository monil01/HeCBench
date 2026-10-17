"""Independent Float64 forward valuation and finite-difference adjoint witness."""
import math,array,json,sys,hashlib
from pathlib import Path
f32=lambda x:array.array('f',[x])[0]
MAT=[4,4,4,8,8,8,20,20,20,28,28,28,40,40,40]
RATES=[f32(x)for x in [.045,.05,.055]*5]
CASES=[(f32(d),f32(l))for d,l in[(.25,.2),(.125,.15),(.375,.25)]]
def value(delta,lam,shift=0):
    L=[f32(.05)]*80;L[-1]+=shift
    for n in range(40):
        cumulative=0.
        for i in range(n+1,80):
            con=delta*lam;cumulative+=con*L[i]/(1+delta*L[i])
            L[i]*=math.exp(con*cumulative+lam*(math.sqrt(delta)*f32(.3)-.5*con))
    discounts=[];annuities=[];b=1.;s=0.
    for x in L[40:]:
        b/=1+delta*x;s+=delta*b;discounts.append(b);annuities.append(s)
    payoffs=[max(0.,100*(1-discounts[m-1]-rate*annuities[m-1]))for m,rate in zip(MAT,RATES)]
    discount=math.prod(1/(1+delta*x)for x in L[:40]);return discount*math.fsum(payoffs)
def main(result_path):
    path=Path(result_path);r=json.loads(path.read_text());expected=[];prefix=[]
    for delta,lam in CASES:
        prefix.extend([lam]*80+[float(m)for m in MAT]+RATES+[delta]);v=value(delta,lam);g=(value(delta,lam,1e-6)-value(delta,lam,-1e-6))/2e-6;half=(value(delta,lam,5e-7)-value(delta,lam,-5e-7))/1e-6
        assert abs(g-half)<1e-5;expected.append({'value':v,'last_rate_greek':g,'half_step_greek':half,'finite_difference_stability':abs(g-half)})
    prefixbytes=array.array('f',prefix).tobytes();records=[]
    for model,m in r['models'].items():
        assert len(m['trials'])==5
        for trial in m['trials']:
            raw=Path(trial['output_artifact']['saved_path']).read_bytes();assert raw[:333*4]==prefixbytes
            output=array.array('f');output.frombytes(raw);assert len(output)==2646
            for c,ref in enumerate(expected):
                offset=333+c*771;prices=output[offset:offset+257];adjointprices=output[offset+257:offset+514];greeks=output[offset+514:offset+771]
                ev=max(abs(x-ref['value'])for x in prices+adjointprices);eg=max(abs(x-ref['last_rate_greek'])for x in greeks);assert ev<=1e-3 and eg<=1e-3;assert all(math.isfinite(x)for x in prices+adjointprices+greeks)
                assert len(set(prices))==len(set(adjointprices))==len(set(greeks))==1
                records.append({'model':model,'trial':trial['trial'],'case':c,'max_price_error':ev,'max_last_initial_rate_greek_error':eg,'forward_vs_adjoint_price_gap':max(abs(x-y)for x,y in zip(prices,adjointprices)),'uniform_paths':True})
    source=Path(__file__).parent/'study_native_source.cuh';native=source.parent.parent/'libor-cuda/main.cu';original=native.read_text();assert source.read_text()==original.replace('#define GRID_SIZE 1500','#define GRID_SIZE 4').replace('#define NPATH 96000','#define NPATH 257')
    report={'result_path':str(path.resolve()),'cases':expected,'all_model_trial_cases':records,'input_prefix_bytes_checked':333*4,'per_artifact_output_values_checked':2313,'artifact_count':20,'native_kernel_bodies_unchanged':True,'native_source_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'native_scope_macros_only':['GRID_SIZE1500to4','NPATH96000to257'],'oracle':'IndependentFloat64forwardLIBORanddiscountedportfolio;centralfinite-differenceoflastinitialrate at h1e-6/5e-7,notthereverseadjointcode.','precision_scope':'ActualFloat32inputconstantsconvertedtoFloat64hostoracle;GPUcompute/storageFloat32;finite-differenceaccuracywitnessnotderivativeofroundedFloat32function.','scientific_scope':'Deterministicidenticalpathsfromfixedshocks. NoMonteCarlosampling/confidenceintervalclaim. OnlylastinitialrateGreekexported,nofullJacobianclaim;named257paths/3marketparameters.','mapping_scope':'HistoricalJuliafamilyglobalperpathscratchpreserved;nativeusesprivatearrays,explicitP1memory/resourceadaptation. Noattributionofgaptoframeworkalone.'}
    out=path.parent/'scientific_price_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main(sys.argv[1])
