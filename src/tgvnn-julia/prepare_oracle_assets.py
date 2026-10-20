#!/usr/bin/env python3
"""Complete CPU oracle package, source-format input RA, scalar SER and layout."""
import pathlib,argparse,json,struct,hashlib
import numpy as np
P=pathlib.Path(__file__).resolve().parent
def main():
 a=argparse.ArgumentParser();a.add_argument('--oracle',type=pathlib.Path,required=True);z=a.parse_args();o=z.oracle;m=json.load(open(o/'oracle_manifest.json'));n=int(np.prod(m['shape']));g=lambda k:np.fromfile(o/(k+'.bin'),dtype='<c16');img=g('img');zero=g('imgz');recon=g('imgr')
 for k,x in [('ser_imgz',zero),('ser_imgr',recon)]:
  v=np.array([-20*np.log10(np.linalg.norm(abs(x)-abs(img))/np.linalg.norm(abs(img)))],dtype='<f8');assert np.isfinite(v).all();p=o/(k+'.bin');v.tofile(p);m['output_assets'][k]={'path':str(p),'dtype':'float64','count':1,'bytes':8,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 for k in ['input_img','input_mask']:
  b=g(k).astype('<c8').tobytes();h=struct.pack('<6Q',0x7961727261776172,0,4,8,len(b),3)+struct.pack('<3Q',*m['shape']);p=o/(k+'.ra');p.write_bytes(h+b);m['input_assets'][k+'_source_format_RA']={'path':str(p),'bytes':len(h+b),'sha256':hashlib.sha256(h+b).hexdigest()}
 order=['img','mask','imgz','imgb','imgl','imgs','w','p','q','r','lbar','sbar','wbar','lold','sold','wold','tmp','imgr','ser_imgz','ser_imgr','ls','ls2'];offset=4*n;rows=[]
 for k in order:
  count=m['output_assets'][k]['count']*(2 if m['output_assets'][k]['dtype']=='complex128'else 1);rows.append({'name':k,'float32_offset':offset,'float32_count':count,'oracle':m['output_assets'][k]});offset+=count
 m.update(serialized_fields=rows,input_prefix_float32_count=4*n,tail_guard_float32_count=68,artifact_float32_count=offset+68,raw_SVD_factor_float32_count=2*(n+2*m['shape'][2]**2),SER_scope='Source magnitude-only norm ratio, not complex reconstruction-error metric or independent clinical quality groundtruth.',completion_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest());(o/'oracle_manifest_completed.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'oracle':str(o),'float32_count':offset+68,'artifact_bytes':4*(offset+68)}))
if __name__=='__main__':main()
