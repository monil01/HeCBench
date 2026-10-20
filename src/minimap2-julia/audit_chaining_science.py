#!/usr/bin/env python3
"""Independent mathematical predecessor-graph/path-score witness on actual native capture."""
import json,struct,hashlib,argparse
from pathlib import Path
P=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def run(result):
 d=json.loads(result.read_text());artifact=Path(d['models']['cuda']['trials'][0]['output_artifact']['saved_path']);raw=artifact.read_bytes();lines=(P/'study_reads.txt').read_text().splitlines();pos=0;pe=0;nodes=edges=seeds=0;longest=0
 while pos<len(lines):
  h=lines[pos].split();n=int(h[0]);avg=f32(float(h[1]));dxmax,dymax,bw=map(int,h[2:]);pos+=1;anchors=[tuple(map(int,s.split())) for s in lines[pos:pos+n]];pos+=n;assert lines[pos]=='EOR';pos+=1;scores=[];parents=[];depth=[]
  for i,(tag,x,w,y) in enumerate(anchors):
   batch,k=divmod(i,1024);offset=d['input_prefix_count']+64+((batch*1792+pe)*1024+k)*8;s,p=struct.unpack_from('<ii',raw,offset);nodes+=1;assert s>=w
   if p==-1:
    assert s==w;seeds+=1;dep=1
   else:
    assert 0<=p<i and i-p<=64;ptag,px,pw,py=anchors[p];dx=x-px;dy=y-py;dd=abs(dx-dy);assert tag==ptag and dx!=0 and dx<=dxmax and 0<dy<=dymax and dd<=bw
    penalty=int(f32(f32(dd)*f32(f32(.01)*avg)));lg=0 if dd<2 else min(8,dd.bit_length()-1);transition=min(dx,dy,w)-penalty-(lg>>1)
    assert s==scores[p]+transition,(pe,i,p,s,scores[p]+transition);edges+=1;dep=depth[p]+1
   scores.append(s);parents.append(p);depth.append(dep);longest=max(longest,dep)
  pe+=1
 report={'benchmark':'minimap2','actual_native_artifact_sha256':hashlib.sha256(raw).hexdigest(),'real_reads_checked':pe,'all_real_anchor_states_checked':nodes,'parent_edges_checked':edges,'seed_nodes_checked':seeds,'longest_parent_path_nodes':longest,'every_score_at_least_anchor_width':True,'every_parent_strictly_earlier_within64':True,'every_parent_tag_and_geometry_admissible':True,'every_actual_path_score_exact_recurrence':True,'acyclic_predecessor_graph':True,'precision':'SourceFloat32 avg/0.01/product rounding then truncationInt32, no tolerance; source log cap8 preserved.','scope':'Source bounded-anchor chaining recurrence/path consistency only; no full read mapper, global unrestricted optimality, biological-alignment quality or end-to-end genome claim. Full oracle checks also cover paddedinactive lanes and carried trackers.'};(P/'chaining_scientific_witness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('result',type=Path);run(ap.parse_args().result.resolve())
