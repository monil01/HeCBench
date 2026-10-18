from pathlib import Path
import struct,json,math,hashlib
p=Path(__file__).resolve().parent;f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];w=lambda x:struct.pack('<I',x);v=lambda x:struct.pack('<f',f32(x));out=bytearray(w(3));cases=[];prefix=1
for ci,(nx,ny)in enumerate([(9,7),(60,46),(13,10)]):
 n2=nx*ny;n3=3*n2;a=[[f32(0)]*(n3 if f<10 else n2)for f in range(15)]
 for j in range(ny):
  for i in range(nx):
   col=i+nx*j;phase=(i+2*j)%5
   for k in range(3):
    q=i+nx*k+nx*3*j
    if ci==0:T=(280.,275.,250.)[k];pi=1.;vapor=1e-6;cloud=0.;rain=0.;snow=0.;den=1.;pres=90000.
    elif ci==1:T=285.+2*k+.125*phase;pi=.95+.005*k;vapor=.012+.0002*((phase+k)%5);cloud=.0005+.0001*k;rain=.0002+.000025*phase;snow=0.;den=1.+.05*k;pres=90000.-5000*k
    else:T=(265.,255.,245.)[k]+.125*phase;pi=(.96,.94,.92)[k];vapor=(.002,.0015,.0009)[k]+.00001*phase;cloud=.0003+.000025*phase;rain=.0001;snow=.0002;den=(1.1,1.,.9)[k];pres=(90000.,80000.,70000.)[k]
    pi=f32(pi);a[0][q]=f32(f32(T)/pi);a[1][q]=pi;a[2][q]=f32(vapor);a[3][q]=f32(cloud);a[4][q]=0.;a[5][q]=f32(rain);a[6][q]=f32(snow);a[7][q]=f32(den);a[8][q]=f32(pres);a[9][q]=500.
   a[10][col]=f32(.125+col/1024)if ci==0 else f32(.05+col/4096);a[11][col]=.001;a[12][col]=.002;a[13][col]=f32(.0625+col/2048)if ci==0 else f32(.025+col/8192);a[14][col]=.003
 for f in range(15):a[f]=list(map(f32,a[f]))
 out +=w(nx)+w(ny)+v(10.)
 for vals in a:
  for x in vals:out +=v(x)
 prefix+=3+sum(map(len,a));cases.append({'case':ci,'nx':nx,'ny':ny,'nz':3,'delt':10.,'kind':['manufactured_dry_clear_identity','nontrivial_warm_cloud_rain','nontrivial_cold_cloud_rain_snow'][ci],'initial_ice_zero':True,'uniform_delz':500.,'fields':['th','pii','q','qc','qi','qr','qs','den','p','delz','rain','rainncv','sr','snow','snowncv'],'layout':'x-fast,thenverticalk,theny;2Dx-fasttheny','state_float_count':sum(map(len,a))})
(p/'fixtures_f32.bin').write_bytes(out);info={'cases':cases,'input_prefix_count':prefix,'output_count':prefix+sum(c['state_float_count']for c in cases)+3*15*128,'fixture_sha256':hashlib.sha256(out).hexdigest(),'dual_reference_tolerance_preGPU':{'atol':1e-7,'rtol':5e-5},'independent_checks_preGPU':{'dry_fixedpoint':'th,q unchanged/pi1/nohydrometeors;rain/snowaccumulatorsunchanged,ncv/sr0','mass_budget':'Initialcolumnsum den*delz*(q+qc+qi+qr+qs) equalsfinal+totalrainincrement; snowissubsetnotdoublecounted','water_rounding_budget':'gamma4096(u32)×(initialwater+abs(initialrain)+1) +3*maxden*delz*qmin; conservativefixed-sourcefiniteoperationbudget, not tunedagainstGPUdrift','source_scope':'Uniformdelz/3levels/initialqi0/mstep1 only, no precipitationtransport-general/energy/climatepredictioncertificate','all_fields_finite':True,'all_water_species_nonnegative':True,'snow_ratio_interval':[0,1],'readonly_and_guards':'allpii/den/p/delzrawbitsunchanged andevery64prefix/suffixguardexact'}};(p/'fixture_manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(prefix,info['output_count'])
