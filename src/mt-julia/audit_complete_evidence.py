#!/usr/bin/env python3
"""CPU-only complete MT raw inputs/outputs/guard joins, no sampling acceptance."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--model',required=True);p.add_argument('--folder',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=Path(a.result);r=json.loads(result.read_text());root=Path(__file__).resolve().parent;dat=(root/'data/MersenneTwister.dat').read_bytes();params=np.frombuffer(dat,dtype='<u4').copy();cases=[(5860,777,3),(38,0,2),(20,0xffffffff,2)];expected=[]
 for c,(n,seed,rep) in enumerate(cases):
  words=params.copy().reshape(4096,4);words[:,3]=seed;expected.extend((c,n,seed&65535,seed>>16,rep,4096,128));expected.extend(words.view(np.uint8).reshape(-1).tolist())
 expected=np.array(expected,dtype='<f4');assert len(expected)==196629;rows=[]
 for t in r['models'][a.model]['trials']:
  trial=t['trial'];path=Path(t['output_artifact']['saved_path']);assert sha(path)==t['output_artifact']['sha256'];x=np.memmap(path,dtype='<f4',mode='r');assert len(x)==48676885 and np.array_equal(x[:196629],expected);offset=196629;moments=[]
  for c,(n,seed,rep) in enumerate(cases):
   count=n*4096;gauss=x[offset:offset+count];uniform=x[offset+count:offset+count*2];assert np.all(np.isfinite(gauss)) and np.all(np.isfinite(uniform)) and np.all((uniform>0)&(uniform<=1));moments.append({'case':c,'count':count,'Gaussian_mean':float(np.mean(gauss,dtype=np.float64)),'Gaussian_second_raw_moment':float(np.mean(gauss.astype(np.float64)**2)),'uniform_min':float(uniform.min()),'uniform_max':float(uniform.max()),'scope':'Descriptivestatisticsnotdistributionacceptancecertificate'});offset+=count*2
  assert offset==len(x);guardpath=Path(a.folder)/f'{a.model}_trial{trial}.guards.bin';raw=guardpath.read_bytes();cursor=0
  for c in range(3):
   for local_id,bytecount in [(0,65536),(1,4096*5860*4)]:
    id,nbytes=struct.unpack_from('<QQ',raw,cursor);cursor+=16;assert id==c*2+local_id and nbytes==bytecount;assert raw[cursor:cursor+512]==bytes([0xa7])*512;cursor+=512
  assert cursor==len(raw)==3168
  metricsfile=Path(a.folder)/f'{a.model}_trial{trial}.audit.jsonl';metrics=[json.loads(s) for s in metricsfile.read_text().splitlines()];assert len(metrics)==3
  for c,row in enumerate(metrics):assert row['case']==c and row['checked']==cases[c][0]*4096*2 and row['uniform_mismatches']==0 and row['gaussian_failures']==0
  rows.append({'trial':trial,'primary_sha256':sha(path),'primary_values':len(x),'actual_input_prefix_exact':True,'actual_guard_bytes':3072,'guards_sidecar_sha256':sha(guardpath),'oracle_receipt_sha256':sha(metricsfile),'oracle':metrics,'moments':moments})
 assert len(rows)==5;out={'benchmark':'mt','model':a.model,'correct':True,'result':str(result),'result_sha256':sha(result),'all5_full_outputs_finite':True,'full_bit_MT_uniform_and_LongDouble_Gaussian_runtime_oracle_pass':True,'parameters_all4096_exact_actual_bytes':True,'input_prefix_values':196629,'primary_values_per_trial':48676885,'trials':rows,'scope':'UniformvaluesareexplicituntimeddeterministicMTreplay, Gaussianactualtimedfinaloutput. Guardsboundphysical96MBoutputbuffer, smallcontrolsuselogicalprefix; no small-logical-endcanaryclaim. Momentsdescriptiveonly.'};Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'correct':True,'trials':5,'full_outputs_per_trial':48480256,'actual_guard_bytes_all5':15360}))
if __name__=='__main__':main()
