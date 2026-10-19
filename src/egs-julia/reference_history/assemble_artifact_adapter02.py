"""Lossless typed EGS capture encoding: integer bytes exact, floating fields numeric."""
from pathlib import Path
from array import array
import json,struct,sys

def assemble(directory,output):
 D=Path(directory);M=json.loads((D/'capture_manifest.json').read_text());desc=M['allocations'];code={'uint32':'I','int32':'i','uint64':'Q','float32':'f','float64':'d'};count=0;prefix=0
 def block(f,v):
  nonlocal count
  a=array('d',iter(v));assert sys.byteorder=='little';a.tofile(f);count+=len(a)
 def encode(f,path,d,initial):
  raw=path.read_bytes();assert len(raw)==d['bytes']
  # Exact external bytes establish identical actualinitialpayloads, includingF32/F64 bits.
  if initial:block(f,raw);return
  name=d['name']
  if name.startswith('stack_'):
   for i in range(32):
    words=raw[16*i:16*i+16]
    if name=='stack_a':block(f,words[:4]);block(f,struct.unpack('<2f',words[4:12]));block(f,words[12:16])
    elif name=='stack_b':block(f,words[:4]);block(f,struct.unpack('<3f',words[4:16]))
    else:block(f,struct.unpack('<4f',words))
  elif d['encoding']in['uint32','int32','uint64']:block(f,raw)
  else:
   values=array(code[d['encoding']]);values.frombytes(raw);block(f,values)
 with Path(output).open('wb')as f:
  for d in desc:encode(f,D/f"initial_{d['name']}.bin",d,True)
  prefix=count
  for cap in M['captures'][1:]:
   for d in desc:
    if d['mutable']:encode(f,D/cap['records'][d['name']]['path'],d,False)
  # Source hostnormalization and totalcategory retained asfullnumerical scientificoutputs, outsideROI.
  grand=0
  for cap in M['captures'][1:]:grand+=struct.unpack('<9Q',(D/cap['records']['step_counts']['path']).read_bytes())[7]
  final=M['captures'][-1]['label'];weight=struct.unpack('<d',(D/f'{final}_total_weight.bin').read_bytes())[0];average=weight/grand;assert average>0
  for kind in ['count','energy']:
   total=array('d',[0.]*(512*512))
   for c in range(4):
    a=array('d');a.frombytes((D/f'{final}_total_{kind}_{c}.bin').read_bytes());a=array('d',(x/average for x in a));block(f,a)
    for i,x in enumerate(a):total[i]+=x
   block(f,total)
 return {'count':count,'input_prefix_count':prefix,'full_normalized_pixels':10*512*512,'encoding':'Externalpayloadbytes andintegerfields asF64 bytevalues; physicalfloatfields typedF64. No rawbitidentity claimforfloatoutputs.'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--capture-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(assemble(a.capture_dir,a.output)))
