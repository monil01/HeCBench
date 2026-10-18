"""Exact scaled-integer complex convolution plus independent delta-kernel identity."""
import argparse,array,math,struct,json,hashlib
from pathlib import Path
F32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
def complex_pairs(values):return list(zip(values[::2],values[1::2]))
def audit(path):
 result=json.loads(Path(path).read_text());cache={};reports=[]
 for model,info in result['models'].items():
  for trial in info.get('trials',[]):
   source=Path(trial['output_artifact']['saved_path']);blob=source.read_bytes();digest=hashlib.sha256(blob).hexdigest()
   if digest not in cache:
    a=array.array('d');a.frombytes(blob);off=0;inputs=[]
    for case in (1,2):
     points=complex_pairs(a[off:off+128]);off+=128;image=complex_pairs(a[off:off+16512]);off+=16512;gcf=complex_pairs(a[off:off+131072]);off+=131072;assert all(x==0 for x in a[off:off+128]);off+=128;assert list(a[off:off+7])==[64,64,32,8,2080,3,case];off+=7;assert all(v==(0,0)for v in image[:2080]+image[2080+4096:]);assert all(x==F32(x)and y==F32(y)for x,y in points);inputs.append((points,image,gcf))
    assert off==295694;cases=[]
    for case,(points,image,gcf)in enumerate(inputs,1):
     actual=list(a[off:off+128]);off+=128;expected=[];bins=set();minimumload=8256;maximumload=-1;outside=0
     for x,y in points:
      mx,my=math.floor(x),math.floor(y);sx,sy=math.floor(8*(x-mx)),math.floor(8*(y-my));assert 0<=sx<8 and 0<=sy<8;bins.add((sx,sy));realnum=imagnum=0
      for dy in range(-16,16):
       for dx in range(-16,16):
        ix,iy=mx+dx,my+dy;address=2080+ix+64*iy;minimumload=min(minimumload,address);maximumload=max(maximumload,address);assert 0<=address<8256
        if not(0<=ix<64 and 0<=iy<64):continue
        vr,vi=image[address];gr,gi=gcf[(sy*8+sx)*1024+(dy+16)*32+dx+16];scaled=[vr*16,vi*16,gr*64,gi*64];assert all(v==round(v)for v in scaled);vr,vi,gr,gi=map(round,scaled);realnum+=vr*gr-vi*gi;imagnum+=vr*gi+vi*gr
      expected.extend([realnum/1024,imagnum/1024])
      if not(0<=mx<64 and 0<=my<64):outside+=1
      if case==2:
       want=image[2080+mx+64*my]if 0<=mx<64 and 0<=my<64 else(0,0);assert expected[-2:]==list(want)
     assert expected==actual;assert len(bins)==64;cases.append({'case':case,'all_64_complex_outputs_exact':True,'exact_integer_scaled_convolution_components':128,'all_subpixel_bins_covered':64,'logical_outside_image_points':outside,'physical_padded_load_range':[minimumload,maximumload],'delta_kernel_identity':case==2})
    assert off==len(a)==295950;cache[digest]=cases
   reports.append({'model':model,'trial':trial['trial'],'artifact_sha256':digest,'cases':cache[digest]})
 return {'benchmark':'degrid','result':str(Path(path).resolve()),'captured_trials_checked':len(reports),'distinct_complete_artifacts_independently_checked':len(cache),'verification_contract':'Fullcomplexoutputs exactscaledintegerconvolution;deltakernelindependentflooredsamples/zerooutside;all64binsandfullpadding/physicaladdressesvalidated.','scope':'FP64complexstorage/arithmetic withactualsourceFloat32floorfbinning;inputcoordinatesexactFloat32dyadics. Nativewarp vsKA/JACCsharedemulation explicitlydifferentresources. NotarbitraryFloat64coordinateprecision,Fourieradjointorfullradioimagingcertificate.','reports':reports}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--out');args=p.parse_args();out=audit(args.result);text=json.dumps(out,indent=2)+'\n';print(text)
 if args.out:Path(args.out).write_text(text)
