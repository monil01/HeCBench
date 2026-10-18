from pathlib import Path
import struct,math,json,hashlib
B=Path(__file__).resolve().parent
GUARD=-1234567.0
CASE_SPECS=[(n,dim,fixture) for n in [257,4099] for dim in [2,5,10] for fixture in [0,1]]
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def objective(xs):return -math.fsum(math.sin(x)*math.sin((i+1)*x*x/math.pi)**20 for i,x in enumerate(xs))
def budget(value,dim):return 3e-5*dim+1e-5*abs(value)
if __name__=='__main__':
 inputs=[];oracles=[];cases=[]
 for n,dim,fixture in CASE_SPECS:
  offset=len(inputs);oracle_offset=len(oracles);values=[]
  for i in range(n):
   xs=[4.0 if fixture else 0.0 if i==0 else f32(math.pi) if i==1 else 4.0 if i==2 else ((i*1973+j*919+dim*337+n*13)%16385)/4096 for j in range(dim)];values+=xs;oracles.append(objective(xs))
  inputs+=values;cases.append({'n':n,'dim':dim,'fixture':fixture,'input_offset':offset,'oracle_offset':oracle_offset,'independent_incumbent_min':min(0.,min(oracles[oracle_offset:])),'meaning':'Seeded incumbent min(0, sampled objectives), not proof of continuous/global minimizer'})
 for name,values,fmt in [('input_f32.bin',inputs,'f'),('oracle_f64.bin',oracles,'d')]:
  (B/name).write_bytes(struct.pack('<'+fmt*len(values),*values))
 description={'cases':cases,'input_prefix_count':sum(n*d+17 for n,d,_ in CASE_SPECS),'complete_capture_count':sum(n*(d+1)+34 for n,d,_ in CASE_SPECS),'source_domain':[0,4],'usual_test_function_domain':'[0,pi], see SFU primarysource; source[0,4] retained, no continuousoptimization certificate','source_incumbent':0,'m':10,'exponent':20,'internal_repeats':100,'fixtures':['Deterministic full Float32 points, zero/pi/4 witnesses anddyadicvariedcoordinates','Allcoordinates4: positiveobjectives, retainedsourceincumbent0, notminoveronlysampledpoints'],'independent_oracle':'Float64 math sin/truepi +fsum, not GPU Float32 expression; tolerance covers sourceFloat32 pi/argument/pow/sum rounding','pointwise_budget':'3e-5*dim+1e-5*abs(oracle)','budget_rationale':'Predeclared finite-case computational target; phaseFloat32 rounding/singleprecisionpi amplified by smooth sin^20 derivative (~2.75 max), plus libm/sum allowance. Not a formal correctly-rounded or allinputs guarantee.','dual_fieldwise_budget':'6e-5*dim+2e-5*maxabs(reference, target); exact inputs/guards and+0incumbent onpositivefixture','timing':'Sum12 case kernel-launch/contextcompletion windows, eachR100 then/timeR; minreset0 once percase outside ROI, noresetbetween internal repeats (sourceprotocol). Full perpointobjectivecapture outside ROI.','source':'https://www.sfu.ca/~ssurjano/michal.html'}
 (B/'fixture_description.json').write_text(json.dumps(description,indent=2)+'\n')
