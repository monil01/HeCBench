#!/usr/bin/env python3
"""Lossless Float32 unit conversion: all complete values retained, invertible bits."""
import pathlib,json,hashlib,argparse
import numpy as np

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def convert(source,layout,units,dest,decode=False):
 raw=np.memmap(source,dtype='<f4',mode='r');assert raw.size==sum(f['count']for f in layout);h=hashlib.sha256();rows=[]
 with open(dest,'wb')as out:
  for f in layout:
   exponent=units['fields'].get(f['name'],{}).get('divide_by_power_of_two_exponent',0);v=raw[f['offset']:f['offset']+f['count']];assert np.isfinite(v).all()
   q=np.ldexp(v,exponent if decode else -exponent).astype('<f4');back=np.ldexp(q,-exponent if decode else exponent).astype('<f4');assert np.isfinite(q).all()and np.array_equal(back.view('<u4'),v.view('<u4')),('nonlossless',f['name']);out.write(q.tobytes());h.update(back.tobytes());rows.append({'field':f['name'],'count':f['count'],'exponent':exponent,'all_original_bits_recovered':True})
 assert digest(source)==h.hexdigest(),'fullbitstreamroundtripmismatch'
 return {'source_path':str(source),'source_sha256':digest(source),'encoded_path':str(dest),'encoded_sha256':digest(dest),'decoded_original_sha256':digest(dest)if decode else h.hexdigest(),'roundtrip_source_sha256':h.hexdigest(),'float32_words':int(raw.size),'all_original_bits_recovered':True,'fields':rows,'representation':'UnchangedrawF32 except4fixedindependentCPU-derived power2unit conversions; nofiltering/phasecanonicalization/checksumsubstitution.','decode':decode}
def main():
 a=argparse.ArgumentParser();a.add_argument('--source',type=pathlib.Path,required=True);a.add_argument('--layout',type=pathlib.Path,required=True);a.add_argument('--units',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--proof',type=pathlib.Path,required=True);a.add_argument('--decode',action='store_true');z=a.parse_args();r=convert(z.source,json.load(open(z.layout)),json.load(open(z.units)),z.output,z.decode);z.proof.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'all_original_bits_recovered':True,'words':r['float32_words'],'proof':str(z.proof)}))
if __name__=='__main__':main()
