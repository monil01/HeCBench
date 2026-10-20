#!/usr/bin/env python3
"""CPU-only source-derived fixtures and independent/original CPU references.
No network, privilege commands, package installation, CUDA context or GPU execution.
"""
import argparse,hashlib,json,struct,subprocess,tempfile
from pathlib import Path

def ppm(path):
 b=path.read_bytes();i=0;words=[]
 while len(words)<4:
  while b[i] in b' \t\n\r':i+=1
  if b[i]==35:i=b.index(10,i)+1;continue
  start=i
  while b[i] not in b' \t\n\r':i+=1
  words.append(b[start:i])
 assert words[0]==b'P6' and words[3]==b'255';w,h=map(int,words[1:3]);i+=1;pixels=b[i:];assert len(pixels)==3*w*h;return w,h,pixels

def main():
 here=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--original-source',default=str(here/'original_native'));p.add_argument('--cxx',default='g++');p.add_argument('--cuda-include',default='/usr/include');p.add_argument('--fixtures-only',action='store_true');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);source=Path(a.original_source);w,h,im0=ppm(source/'data/frame10.ppm');w1,h1,im1=ppm(source/'data/frame11.ppm');assert(w,h,w1,h1)==(640,480,640,480)
 def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
 for name,cw,ch,s,levels,left,right in [('original_pair',640,480,768,5,im0,im1),('identity_partial',65,49,256,3,im0,im0)]:
  values=[]
  for image in(left,right):
   for y in range(ch):values.extend([f32(image[(y*640+x)*3]/255.)for x in range(cw)]+[0.]*(s-cw))
  raw=struct.pack('<8If',0x4f463537,cw,ch,s,levels,3,500,len(values),f32(.2))+struct.pack('<%df'%len(values),*values);(out/f'{name}_fixture.bin').write_bytes(raw)
 if not a.fixtures_only:
  with tempfile.TemporaryDirectory(prefix='opticalFlow-cpu-')as temp:
   temp=Path(temp)
   subprocess.run([a.cxx,'-O3','-std=c++17',str(here/'independent_flow_reference.cpp'),'-o',str(temp/'independent')],check=True)
   subprocess.run([a.cxx,'-O3','-std=c++17','-I'+a.cuda_include,'-include','cuda_runtime.h','-x','c++',str(here/'generate_legacy_CPU_reference.cpp'),str(source/'flowGold.cu'),'-o',str(temp/'legacy')],check=True)
   subprocess.run([a.cxx,'-O3','-std=c++17',str(here/'prepare_primitive_fixture.cpp'),'-o',str(temp/'primitive')],check=True)
   for name in('original_pair','identity_partial'):
    subprocess.run([str(temp/'independent'),str(out/f'{name}_fixture.bin'),str(out/f'{name}_oracle.bin')],check=True)
    subprocess.run([str(temp/'legacy'),str(out/f'{name}_fixture.bin'),str(out/f'{name}_legacyCPU.bin')],check=True)
   subprocess.run([str(temp/'primitive'),str(out/'primitive_fixture.bin'),str(out/'primitive_oracle.bin')],check=True)
 print(json.dumps({'CPU_only':True,'fixtures_only':a.fixtures_only,'assets':[{'file':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in sorted(out.glob('*.bin'))]},indent=2))
if __name__=='__main__':main()
