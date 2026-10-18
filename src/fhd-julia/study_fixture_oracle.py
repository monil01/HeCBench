from pathlib import Path
import math,struct,json
p=Path(__file__).resolve().parent
f=[];o=[];cases=[]
for ns,nv,mode in[(17,31,0),(33,1,1),(257,4097,2)]:
 r=[((j*13%17)-8)/256 for j in range(nv)];im=[((j*7%19)-9)/256 for j in range(nv)];k=[tuple(0. if mode==0 else((j*11+dim*7)%17-8)/32 for dim in range(3))for j in range(nv)]
 if mode==1:r=[.375];im=[-.125];k=[(.25,-.125,.5)]
 a=[(j%11-5)/128 for j in range(ns)];b=[(j%7-3)/256 for j in range(ns)];x=[(j*3%17-8)/16 for j in range(ns)];y=[(j*5%19-9)/16 for j in range(ns)];z=[(j*7%13-6)/16 for j in range(ns)]
 rr=[];ii=[]
 for j in range(ns):
  terms=[complex(r[m],im[m])*complex(math.cos(2*math.pi*math.fsum([k[m][0]*x[j],k[m][1]*y[j],k[m][2]*z[j]])),math.sin(2*math.pi*math.fsum([k[m][0]*x[j],k[m][1]*y[j],k[m][2]*z[j]])))for m in range(nv)];rr.append(a[j]+math.fsum(v.real for v in terms));ii.append(b[j]+math.fsum(v.imag for v in terms))
 f +=[ns,nv,mode]+r+im+[v for xyz in k for v in xyz]+a+b+x+y+z;o+=rr+ii;cases.append({'samples':ns,'voxels':nv,'mode':mode,'chunks':(nv+4095)//4096,'max_abs_output':max(map(abs,rr+ii)),'oracle':'Float64complexphasor+fsum, exactdyadicinputs; zero-frequency/single-modeanalyticcontrols','phase_rounding_bound':2e-6,'intrinsic_SFU_absolute_error_budget':2e-6,'accumulation_contract':'Prospectivefullcomponentatol1e-4 rtol2e-5; sourceorderedFloat32 sum, not Float64storage. Also report theoreticalforwardbound gamma(4*nv+16)*sum_abs_components + phase_and_SFU_budgets*sum_abs_amplitudes (maylooser thanacceptance)','gamma4N_plus16':(4*nv+16)*2**-24/(1-(4*nv+16)*2**-24)})
(p/'fixtures_f32.bin').write_bytes(struct.pack('<%sf'%len(f),*f));(p/'oracle_f64.bin').write_bytes(struct.pack('<%sd'%len(o),*o));(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent Python complex exponential sum with real2pi and Float64fsum; analyticzero-frequency andsingle-modecontrols; sourcepositiveexponential convention','cases':cases,'input_prefix_count':len(f),'oracle_values':len(o),'acceptance':{'atol':1e-4,'rtol':2e-5,'declared_before_GPU':True,'SFU_scope':'Inputphasebounded<=~2.3radians ondyadicfixtures, native __sinf/__cosf andJulia Base.FastMath methods use sameCUDAfastmathbackend; no broadlargeargument guarantee'}},indent=2)+'\n')
