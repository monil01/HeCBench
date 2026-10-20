#!/usr/bin/env python3
"""Exact actual native/Julia device input bytes, complete artifacts, guard joins."""
import argparse,hashlib,json,struct,ctypes
from pathlib import Path
import numpy as np
from audit_mathematical_outputs import physical,read_records,LENGTH

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--model',required=True);p.add_argument('--folder',required=True);p.add_argument('--source-root',default=str(Path(__file__).resolve().parent));p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.source_root);r=json.loads(Path(a.result).read_text());m=r['models'][a.model];folder=Path(a.folder)
 initial=np.fromfile(root/'source_initial_parameters.bin',dtype='<f4');case=root/'cases/balanced100train20test';images=[];labels=[]
 for prefix in ('train','t10k'):
  raw=(case/f'{prefix}-images.idx3-ubyte').read_bytes();tag,n,h,w=struct.unpack_from('>4I',raw);assert(tag,h,w)==(2051,28,28);images.append(np.frombuffer(raw,dtype='u1',offset=16).astype(np.float64).reshape(n,784)/255.)
  raw=(case/f'{prefix}-labels.idx1-ubyte').read_bytes();tag2,n2=struct.unpack_from('>2I',raw);assert tag2==2049 and n2==n;labels.append(np.frombuffer(raw,dtype='u1',offset=8).astype(np.float64))
 expected=np.concatenate([initial.astype(np.float64),*labels,images[0].astype(np.float32).astype(np.float64).ravel(),images[0].astype(np.float32).astype(np.float64).ravel(),images[1].astype(np.float32).astype(np.float64).ravel()]);assert len(expected)==174943
 rows=[]
 for t in m['trials']:
  trial=t['trial'];art=t['output_artifact'];path=Path(art['path']);
  if not path.is_file():path=Path(a.result).parent/art['path']
  x=np.fromfile(path,dtype='<f8');assert len(x)==8981929 and np.array_equal(x[:174943],expected)
  science=folder/f'{a.model}_trial{trial}.science.bin';phys=folder/f'{a.model}_trial{trial}.physical.bin';counts={0:0,1:0,2:0};forwardimages=[]
  for header,ls in read_records(science):
   kind,step,label,pred=header;counts[kind]+=1
   if kind in(0,2):
    expectedimage=images[0 if kind==0 else 1][step].astype(np.float32).astype(np.longdouble);assert np.array_equal(ls[0]['output'],expectedimage);assert label==int(labels[0 if kind==0 else 1][step]);forwardimages.append(1)
  assert counts=={0:200,1:200,2:20}
  rows.append({'trial':trial,'actual_input_prefix_values':174943,'primary_values':len(x),'primary_sha256':sha(path),'science_sha256':sha(science),'actual_image_uploads':len(forwardimages),'physical':physical(phys)})
 assert len(rows)==5
 # Independent glibc default-rand1 provenance witness, not GPU initializer replacement.
 libc=ctypes.CDLL(None);libc.srand.argtypes=[ctypes.c_uint];libc.rand.restype=ctypes.c_int;libc.srand(1);expectedparams=[]
 for width,n in ((0,0),(25,6),(16,1),(216,10)):
  b=[];w=[]
  def rnd():return np.float32(np.float32(.5)-np.float32(np.float32(libc.rand())/np.float32(2147483647)))
  for i in range(n):b.append(rnd());w.extend(rnd() for _ in range(width))
  expectedparams+=b+w
 assert np.array_equal(initial.view(np.uint32),np.array(expectedparams,dtype=np.float32).view(np.uint32))
 out={'benchmark':'mnist','model':a.model,'correct':True,'result':a.result,'result_sha256':sha(Path(a.result)),'actual_input_prefix_exact':True,'default_glibc_rand1_matches_actual_original_ctor_parameter_bits':True,'parameters':len(initial),'guards_bytes_per_trial':28*512,'trials':rows,'scope':'Everyactualinitialparameter/GPUimage/hostlabel and fullactualGPUguardbytes checked; inputunusedderivativepayloadrawonly, notclaimedinitializednumericdata.'};Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'correct':True,'trials':len(rows),'actual_input_prefix':len(expected),'guards_per_trial':28*512}))
if __name__=='__main__':main()
