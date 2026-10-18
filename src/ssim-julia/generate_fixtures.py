from pathlib import Path
import numpy as np,json,hashlib,ctypes
P=Path(__file__).resolve().parent;libc=ctypes.CDLL(None);libc.srand.argtypes=[ctypes.c_uint];libc.rand.restype=ctypes.c_int;W=7;NP=343;cov=np.float32(np.float32(NP)/np.float32(NP-1));K1=np.float32(.01);K2=np.float32(.03);C1=float(np.float32(K1*K1));C2=float(np.float32(K2*K2));cases=[];tiles=[];inputs=[];oracles=[];quality=[]
def boxsum(a):
 p=np.pad(a,((1,0),(1,0),(1,0))).cumsum(0).cumsum(1).cumsum(2)
 return(p[W:,W:,W:]-p[:-W,W:,W:]-p[W:,:-W,W:]-p[W:,W:,:-W]+p[:-W,:-W,W:]+p[:-W,W:,:-W]+p[W:,:-W,:-W]-p[:-W,:-W,:-W])
for ci,(dims,mode)in enumerate([((67,41,37),'source_scaled_random'),((15,15,15),'identical_half'),((17,17,17),'complement_ramp'),((7,7,7),'black_identity'),((7,7,7),'black_white')]):
 batch=tuple(min(v,b)for v,b in zip(dims,(4096,16,16)));bg=tuple(v+6 for v in batch);gcount=((np.prod(bg)+255)//256)*256;gcount=int(gcount)
 if mode=='source_scaled_random':
  libc.srand(123);x=np.array([np.float32(libc.rand())/np.float32(2147483647)for _ in range(gcount)],dtype='<f4');y=(x*np.float32(.75)).astype('<f4')
 elif mode=='identical_half':x=np.full(gcount,.5,dtype='<f4');y=x.copy()
 elif mode=='complement_ramp':
  i=np.arange(gcount);x=(((i%bg[0])+2*((i//bg[0])%bg[1])+3*(i//(bg[0]*bg[1])))%64).astype('f4')/64;y=(np.float32(1)-x).astype('<f4')
 elif mode=='black_identity':x=np.zeros(gcount,dtype='<f4');y=x.copy()
 else:x=np.zeros(gcount,dtype='<f4');y=np.ones(gcount,dtype='<f4')
 io=sum(len(a)for a in inputs);inputs.extend([x.view('<u4'),y.view('<u4')]);case=dict(case=ci,dims=list(dims),batch=list(batch),batch_grid=list(bg),gcount=gcount,output_capacity=3*gcount,input_offset=io,mode=mode,tile_begin=len(tiles));sample=[]
 for oz in range(3,dims[2]-3,batch[2]):
  for oy in range(3,dims[1]-3,batch[1]):
   for ox in range(3,dims[0]-3,batch[0]):
    block=tuple(min(b,d-3-o)for b,d,o in zip(batch,dims,(ox,oy,oz)));grid=tuple(v+6 for v in block);cnt=int(np.prod(block));glen=int(np.prod(grid));X=x[:glen].astype('f8').reshape(grid[::-1]);Y=y[:glen].astype('f8').reshape(grid[::-1]);ux=boxsum(X)/NP;uy=boxsum(Y)/NP;vx=float(cov)*(boxsum(X*X)/NP-ux*ux);vy=float(cov)*(boxsum(Y*Y)/NP-uy*uy);vxy=float(cov)*(boxsum(X*Y)/NP-ux*uy);S=((2*ux*uy+C1)*(2*vxy+C2))/((ux*ux+uy*uy+C1)*(vx+vy+C2));flat=S.ravel();assert len(flat)==cnt and np.isfinite(flat).all();oo=sum(len(a)for a in oracles);oracles.append(flat);tiles.append(dict(case=ci,block=list(block),grid=list(grid),offset=[ox,oy,oz],count=cnt,oracle_offset=oo,oracle_sum=float(flat.sum()),oracle_min=float(flat.min()),oracle_max=float(flat.max())))
    if mode=='source_scaled_random':
     a=.75;ideal=((2*a*ux*ux+C1)/((1+a*a)*ux*ux+C1))*((2*a*vx+C2)/((1+a*a)*vx+C2));sample.append(float(np.max(np.abs(S-ideal))))
    elif mode in['identical_half','black_identity']:assert np.max(np.abs(S-1))<1e-12
    elif mode=='black_white':assert np.max(np.abs(S-C1/(1+C1)))<1e-12
 case['tile_end']=len(tiles);case['valid_volume_count']=int(np.prod(np.array(dims)-6));assert sum(t['count']for t in tiles[case['tile_begin']:case['tile_end']])==case['valid_volume_count'];case['oracle_normalized']=sum(t['oracle_sum']for t in tiles[case['tile_begin']:case['tile_end']])/case['valid_volume_count'];cases.append(case);quality.append(dict(case=ci,mode=mode,identity_analytic_S=1 if mode in['identical_half','black_identity']else None,black_white_analytic_S=C1/(1+C1)if mode=='black_white'else None,scaled_relationship_deviation_due_to_input_rounding=max(sample)if sample else None))
inp=np.concatenate(inputs);out=np.concatenate(oracles);(P/'input_u32.bin').write_bytes(inp.tobytes());(P/'oracle_f64.bin').write_bytes(out.tobytes());prefix=len(inp)+16*len(cases)+12*len(tiles);values=len(out)+2*len(tiles)+len(cases);guards=sum((c['output_capacity']+16-t['count'])for c in cases for t in tiles[c['tile_begin']:c['tile_end']])+32*len(cases);meta=dict(NumPy=np.__version__,cases=cases,tiles=tiles,parameters=dict(W=7,NP=343,cov_norm=float(cov),K1=float(K1),K2=float(K2),data_range=1,C1=C1,C2=C2),input_words=len(inp),oracle_count=len(out),prefix_count=prefix,output_count=values,guard_count=guards,artifact_count=prefix+values+guards,atol=2e-5,rtol=3e-5,independent_quality=quality,oracle='CPUFloat64three-dimensionalprefix-sum-boxmoments, independentofGPU343sampleinnerloop; actualFP32input/constantbits, samplecovariance; fulllocalS/sums/averages plusidentity/complement/gaincontrols.',source_scope='Namedfivevolumescases, uniformW7/samplenorm/crop/sourcebatchmin/partialtileplanning andsamefixedbatchinputreuse; notcoherentoriginal4096×1024²fullvolumetricinput orproductionthroughput. OriginalsourceR1 preserved.',fixtures={n:dict(sha256=hashlib.sha256((P/n).read_bytes()).hexdigest(),bytes=(P/n).stat().st_size)for n in['input_u32.bin','oracle_f64.bin']});(P/'study_plan.json').write_text(json.dumps(meta,indent=2)+'\n')
h=['struct Case{int dx,dy,dz,bx,by,bz,gcount,cap,io,tbegin,tend,voxels;};','struct Tile{int ci,bx,by,bz,gx,gy,gz,ox,oy,oz,count,oo;};','const Case cases[]={']
for c in cases:h.append('{'+','.join(map(str,[*c['dims'],*c['batch'],c['gcount'],c['output_capacity'],c['input_offset'],c['tile_begin'],c['tile_end'],c['valid_volume_count']]))+'},')
h+=['};','const Tile tiles[]={']
for t in tiles:h.append('{'+','.join(map(str,[t['case'],*t['block'],*t['grid'],*t['offset'],t['count'],t['oracle_offset']]))+'},')
h+=['};'];(P/'study_cases.h').write_text('\n'.join(h)+'\n');j=['const CASES=[']
for c in cases:
 d=dict(zip(['dx','dy','dz','bx','by','bz'],[*c['dims'],*c['batch']]));d.update(gcount=c['gcount'],cap=c['output_capacity'],io=c['input_offset'],tbegin=c['tile_begin'],tend=c['tile_end'],voxels=c['valid_volume_count']);j.append('('+','.join(k+'='+str(v)for k,v in d.items())+',),')
j+= [']','const TILES=[']
for t in tiles:
 d=dict(ci=t['case'],**dict(zip(['bx','by','bz','gx','gy','gz','ox','oy','oz'],[*t['block'],*t['grid'],*t['offset']])),count=t['count'],oo=t['oracle_offset']);j.append('('+','.join(k+'='+str(v)for k,v in d.items())+',),')
j.append(']');(P/'study_cases.jl').write_text('\n'.join(j)+'\n');print('SSIMPLAN',len(cases),'cases',len(tiles),'tiles',len(out),'windows',prefix,values,guards,meta['artifact_count'])
