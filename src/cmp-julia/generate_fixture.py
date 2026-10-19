"""Known synthetic seismic common-midpoint controls; no external seismic data."""
from pathlib import Path
import struct,math,json,hashlib
P=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
rows=[]
for ci,(ns,nc,counts,itau)in enumerate([(257,5,[5,4,3,4],.002),(1025,9,[16,12,8,16],.0141)]):
 c0=f32(f32(1.98e-7)*1e6);c1=f32(f32(1.77e-6)*1e6);inc=f32(f32(c1-c0)*f32(1/nc));dt=f32(.002);idt=f32(1/dt);tau=int(f32(f32(itau)*idt));w=2*tau+1;assert w<=16
 trueindex=nc//2;truec=f32(c0+f32(inc*trueindex));tzero=.2 if ci==0 else .6;alltraces=[];blob=bytearray();traceid=0
 for group,count in enumerate(counts):
  for tr in range(count):
   scale=[0,2,-2,1][tr%4];effective=1 if scale==0 else scale if scale>0 else 1/scale
   half=32+tr*16 if ci==0 else 32+tr*32
   # Exact integer raw coordinates and Float32 source half-offset math.
   rawhalf=int(half/abs(effective));sx=-rawhalf;gx=rawhalf;sy=-4*tr;gy=4*tr
   hx=f32(f32(gx-sx)*f32(effective));hy=f32(f32(gy-sy)*f32(effective));h=f32(float(f32(.25*f32(f32(hx*hx)+f32(hy*hy))))/1e6)
   assert math.sqrt(h*1e6)<=600
   event=math.sqrt(tzero*tzero+truec*h);samples=[]
   for t in range(ns):
    if group==0:
     z=(t*float(dt)-event)/.016;v=(1-2*z*z)*math.exp(-z*z)
    elif group==1:v=.25
    elif group==2:v=0.
    else:v=.5 if tr%2==0 else -.5
    samples.append(f32(v))
   head=bytearray(240);traceid+=1
   for off,value in [(0,traceid),(4,traceid),(20,100+group),(24,tr+1),(72,sx),(76,sy),(80,gx),(84,gy)]:struct.pack_into('<i',head,off,value)
   struct.pack_into('<h',head,70,scale);struct.pack_into('<H',head,114,ns);struct.pack_into('<H',head,116,2000)
   blob+=head+struct.pack('<'+'f'*ns,*samples);alltraces.append({'cdp':100+group,'scalco':scale,'sx':sx,'sy':sy,'gx':gx,'gy':gy,'h_source_scalar_check':h,'samples':samples})
 # One actualsource-aperture rejection control, notpart of computationaltracearrays.
 head=bytearray(240);struct.pack_into('<i',head,20,999);struct.pack_into('<i',head,72,-1500);struct.pack_into('<i',head,80,1500);struct.pack_into('<H',head,114,ns);struct.pack_into('<H',head,116,2000)
 blob+=head+bytes(ns*4)
 path=P/f'synthetic{ci}.su';path.write_bytes(blob)
 rows.append({'case':ci,'ns':ns,'nc':nc,'counts':counts,'ncdps':4,'ttraces':sum(counts),'max_padded_traces':max(counts)+(max(counts)&1),'c0':c0,'c1':c1,'inc':inc,'dt':dt,'idt':idt,'itau':f32(itau),'tau':tau,'window':w,'true_event_c_index':trueindex,'true_event_c_scaled':truec,'zero_offset_event_seconds':tzero,'aperture':600,'one_trace_rejected_by_original_aperture':True,'SU':path.name,'SU_sha256':hashlib.sha256(blob).hexdigest(),'SU_bytes':len(blob),'traces':alltraces})
(P/'synthetic_definition.json').write_text(json.dumps({'cases':rows,'scope':'FullysyntheticRickerhyperbola/constant/zero/opposite-amplitude controls; actualsourceSUloader/aperture/CDPgroupingwillbevalidated. Sourcewindowbound<=16 explicit, nooriginal81MBDVCdatasetclaim.'},indent=2)+'\n')
print([{k:v for k,v in q.items()if k!='traces'}for q in rows])
