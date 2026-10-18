from pathlib import Path
import struct,json,math
p=Path(__file__).resolve().parent;fixture=[];gold=[];arrays=[];paths=[]
for ty in range(1,5):
 for dim,nx,ny in [(1,257,1),(2,17,31),(2,1,19)]:
  n=nx*ny;vals=[((i*17+i//5)%37)-18 for i in range(n)];vals[:5]=[9,9,1,0,-2];vals[-2:]=[20,20]
  if ty==2:vals[7]=2**63-1;vals[13]=-2**63
  if ty>=3:
   vals=[float(x)+.25 for x in vals];vals[7]=float('nan');vals[13]=float('inf');vals[9]=-0.0;vals[10]=0.0
   if n>31:vals[31]=-float('inf')
  fmt=['i','q','f','d'][ty-1];ufmt='I' if ty in [1,3] else 'Q';bits=[struct.unpack('<'+ufmt,struct.pack('<'+fmt,v))[0] for v in vals]
  vals=[struct.unpack('<'+fmt,struct.pack('<'+ufmt,v))[0] for v in bits]
  fixture += [ty,dim,nx,ny]+bits
  ai=len(arrays);arrays.append(dict(type=ty,dimension=dim,nx=nx,ny=ny,n=n))
  combos=[(o,True) for o in [1,2,4,8,16,32,64,128]]+[(o,False) for o in [1,8,16]] if nx!=1 else [(1,True),(1,False)]
  for axis in ([1] if dim==1 else [0,1]):
   for order,clip in combos:
    mask=[]
    def at(coord,length):return min(max(coord,0),length-1) if clip else coord%length
    for j in range(n):
     row,col=divmod(j,nx);v=vals[j];ok=True
     for o in range(1,order+1):
      if dim==1:plus=at(j+o,n);minus=at(j-o,n)
      elif axis==0:plus=at(row+o,ny)*nx+col;minus=at(row-o,ny)*nx+col
      else:plus=row*nx+at(col+o,nx);minus=row*nx+at(col-o,nx)
      ok &= v>vals[plus] and v>=vals[minus]
     mask.append(int(ok))
    gb=0xA5A5A5A5 if ty in [1,3] else 0xA5A5A5A5A5A5A5A5
    gold += [165]*8+mask+[165]*8+[gb]*8+bits+[gb]*8
    paths.append(dict(array=ai,type=ty,dimension=dim,nx=nx,ny=ny,n=n,axis=axis,order=order,clip=clip,true_count=sum(mask),output_words=2*(n+16)))
assert len(paths)==148
p.joinpath('fixture.bin').write_bytes(struct.pack('<%sQ'%len(fixture),*fixture));p.joinpath('oracle.bin').write_bytes(struct.pack('<%sQ'%len(gold),*gold));p.joinpath('fixture_description.json').write_text(json.dumps(dict(arrays=arrays,paths=paths,input_prefix_count=len(fixture),output_count=len(gold),oracle='Independent coordinate-neighborhood all-comparison oracle; strictforward >, nonstrictbackward >=; clipclamp or mathmodulo wrap(order<=axislength). Rawtypedbits includingNaN/Inf/signedzero/Int64extremes preserved.'),indent=2))
