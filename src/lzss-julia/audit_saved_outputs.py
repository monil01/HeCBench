#!/usr/bin/env python3
"""Rehash and stream-compare all20 complete captures against independently decoded CPU control."""
from pathlib import Path
import argparse,hashlib,json
P=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('result',type=Path);a.add_argument('--output',type=Path,required=True);x=a.parse_args();d=json.load(open(x.result));assert d['matched_performance_valid']and d['reference_agreement'];rows=[]
 for model,m in d['models'].items():
  assert m['complete']and m['timing_valid']and m['provenance_valid']and len(m['trials'])==5
  for trial in m['trials']:
   art=trial['output_artifact'];f=Path(art['saved_path']);assert sha(f)==art['sha256']and f.stat().st_size==156165962
   with open(f,'rb')as actual:
    for name in ['oracle_prefix_u8.bin','oracle_output_u8.bin']:
     with open(P/name,'rb')as expect:
      while b:=expect.read(4<<20):assert actual.read(len(b))==b,(model,trial['trial'],name)
    assert actual.read(1)==b''
   rows.append(dict(model=model,trial=trial['trial'],path=str(f),bytes=f.stat().st_size,sha256=art['sha256'],full_saved_bytes_exact=True))
 assert len(rows)==20
 proof=dict(result=str(x.result),result_sha256=sha(x.result),captures=rows,all20_full_byte_exact=True,checked_bytes_each=156165962,total_saved_capture_bytes=20*156165962,input_prefix_bytes=50020544,output_bytes=106145418,independent_control='generate_oracle.cpp independentlystructuredstreamingdecoder enforcestokenbounds/consumption andcompletebitroundtrip. Scalarencoder reproducessourcechoice; nooptimal-LZ77claim.',finite_float_check_not_applicable='AllvaluesUInt8artifacts ofuninterpretedUInt32bits; IEEEpatternsarebyteexact, nofloatingarithmetic.',extra_GPU_executions=0)
 x.output.write_text(json.dumps(proof,indent=2)+'\n');print('AUDIT',len(rows),'completecaptures',len({z['sha256']for z in rows}),'distinctSHA',proof['total_saved_capture_bytes'])
if __name__=='__main__':main()
