from pathlib import Path
import json,math,struct,hashlib
P=Path(__file__).resolve().parent
B,C,OH,OW,K,IW,IH=2,3,19,23,9,37,29
f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
image=[-11.125]*(B*C*(IW+1)*(IH+1))
for b in range(B):
 for c in range(C):
  for y in range(IH):
   for x in range(IW):image[((b*C+c)*IH+y)*IW+x]=b*8+c*2+x/16+y/8+x*y/1024
nk=B*K*OH*OW;weights=[0.]*nk;hor=[0.]*nk;ver=[0.]*nk
for b in range(B):
 for y in range(OH):
  for x in range(OW):
   w=[1+(b*7+k*3+y*5+x)%9 for k in range(K)];s=sum(w)
   for k in range(K):
    i=((b*K+k)*OH+y)*OW+x;weights[i]=f32(w[k]/s);hor[i]=((b+2*k+3*y+x)%17-8)/4;ver[i]=((2*b+k+y+3*x)%17-8)/4
expected=[];left=right=top=bottom=0;partition=0.
for b in range(B):
 for c in range(C):
  for y in range(OH):
   for x in range(OW):
    result=0.;partition=max(partition,abs(sum(weights[((b*K+k)*OH+y)*OW+x]for k in range(K))-1))
    for k in range(K):
     i=((b*K+k)*OH+y)*OW+x;px=(x+.5)/OW*(IW-2)+k%3+hor[i]-.5;py=(y+.5)/OH*(IH-2)+k//3+ver[i]-.5;fx=math.floor(px);fy=math.floor(py);xl=max(0,min(fx,IW-1));xr=max(0,min(xl+1,IW-1));yt=max(0,min(fy,IH-1));yb=max(0,min(yt+1,IH-1));xe=xl+(px-fx)*(xr-xl);ye=yt+(py-fy)*(yb-yt);result+=(b*8+c*2+xe/16+ye/8+xe*ye/1024)*weights[i];left+=px<0;right+=px>=IW;top+=py<0;bottom+=py>=IH
    expected.append(result)
blob=image+weights+hor+ver;(P/'study_inputs_f32.bin').write_bytes(struct.pack('<'+str(len(blob))+'f',*blob));(P/'oracle_f64.bin').write_bytes(struct.pack('<'+str(len(expected))+'d',*expected))
meta=dict(generator='generate_oracle.py',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),params=[B,C,OH,OW,K,IW,IH],partitions=[len(image),nk,nk,nk],input_sha256=hashlib.sha256((P/'study_inputs_f32.bin').read_bytes()).hexdigest(),oracle_sha256=hashlib.sha256((P/'oracle_f64.bin').read_bytes()).hexdigest(),max_partition_unity_error=partition,source_edge_samples=dict(left=left,right=right,top=top,bottom=bottom),oracle_range=[min(expected),max(expected)],layout_witness={'coordinate':[0,0,1,0],'native_rowstride_index':IW,'historical_julia_rowstride_index':IH,'native_value':image[IW],'historical_value':image[IH],'reason':'Nativey*image_w, Juliaidx_imgusedy*image_h; defaultsquaremaskedbug.'});(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
