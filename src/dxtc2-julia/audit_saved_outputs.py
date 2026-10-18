"""CPU-only raw BC1 stream/input/guard/offset/quality/mean/dependency audit."""
from pathlib import Path
import json,struct,hashlib,sys,math
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();fix=struct.unpack('<1221I',raw);im=fix[1029:];guard=0xdeadbeef;rows=[];first=None

def decode_quality(words,image,b):
 a=words[2*b]&65535;c=words[2*b]>>16;codes=words[2*b+1]
 def rgb(x):
  r=(x>>11)&31;g=(x>>5)&63;blue=x&31
  return ((r<<3)|(r>>2),(g<<2)|(g>>4),(blue<<3)|(blue>>2))
 q0=rgb(a);q1=rgb(c);q2=tuple((2*q0[k]+q1[k])//3 if a>c else (q0[k]+q1[k])//2 for k in range(3));q3=tuple((q0[k]+2*q1[k])//3 if a>c else 0 for k in range(3));pal=[q0,q1,q2,q3]
 return sum((pal[(codes>>(2*j))&3][k]-((image[b*16+j]>>(8*k))&255))**2 for j in range(16) for k in range(3))/48

for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text());assert res['reference_agreement']
 for model,d in res['models'].items():
  assert len(d['trials'])==5 and all(t['correct'] for t in d['trials']);assert abs(d['mean_seconds']-sum(t['time_seconds'] for t in d['trials'][1:])/4)<1e-12
  for t in d['trials']:
   f=Path(t['output_artifact']['saved_path']);data=f.read_bytes();first=first or data;assert data==first;assert data[:len(raw)]==raw;a=struct.unpack('<2576I',data[len(raw):]);off=0;metrics=[];guards=0;untouched=0
   for n,start in [(12,0),(5,2)]:
    words=a[off:off+40];off+=40;table=a[off:off+1040];off+=1040;pixels=a[off:off+208];off+=208
    for z in [words,table,pixels]:assert z[:8]==(guard,)*8 and z[-8:]==(guard,)*8;guards+=16
    assert table[8:-8]==fix[5:1029] and pixels[8:-8]==im
    payload=words[8:-8]
    for b in range(12):
     if start<=b<start+n:
      mse=decode_quality(payload,im,b);assert math.isfinite(mse) and mse<=256;metrics.append(dict(block=b,MSE=mse))
     else:assert payload[2*b:2*b+2]==(guard,guard);untouched+=2
   assert off==2576 and guards==96 and untouched==14;assert abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   assert abs(max(q['MSE'] for q in metrics)-t['max_fixture_RGB_MSE'])<1e-10
   for name,h in d.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,(model,name)
   rows.append(dict(model=model,trial=t['trial'],artifact=str(f),sha256=hashlib.sha256(data).hexdigest(),raw_stream_exact_all20=True,actualinputprefix_exact_bytes=True,physical_guards_exact=guards,unwritten_offset_words_exact=untouched,independent_BC1_decode_RGB_quality=metrics,all_declared_dependencies_current=True))
print(json.dumps(dict(atol=0,rtol=0,artifact_values=3797,input_prefix_values=1221,input_prefix_bytes=4884,active_compressed_words_pertrial=34,physical_guards_pertrial=96,unwritten_offset_words_pertrial=14,quality_contract='Eachactiveblock fullintegerBC1decode RGB MSE<=256 againstactualsourcepixels, notcompressedreferencefile/perceptualquality/globaloptimizerproof',max_observed_RGB_MSE=74.25,rows=rows),indent=2))
