#!/usr/bin/env python3
"""DNA-derived canonical keys and independent complete integer count oracle."""
from pathlib import Path
from collections import Counter
import random,struct,json,hashlib
P=Path(__file__).resolve().parent
CASES=[(8,1,33,128,3,7),(28,2,257,1024,3,37),(60,4,4099,2048,16,257),(92,6,4099,2048,16,257),(135,10,30000,4096,30,1021)]
def main():
 blob=bytearray(struct.pack('<I',len(CASES)));meta=[];prefix=0;outputs=0
 for k,ip,n,entries,threshold,families in CASES:
  rng=random.Random(20261019+k);seqs=[''.join(rng.choice('ACGT') for _ in range(k)) for _ in range(families)];tr=str.maketrans('ACGT','TGCA');seqs=[min(s,s.translate(tr)[::-1]) for s in seqs]
  ints=[]
  for s in seqs:
   v=0
   for base in s:v=(v<<2)|'ACGT'.index(base)
   v<<=ip*32-2*k;ints.append(tuple((v>>(32*j))&0xffffffff for j in range(ip)))
  keys=[ints[0] if k==135 and i<2000 else ints[(i*13+i//families)%families] for i in range(n)];counts=Counter(keys);flat=[v for key in keys for v in key];ordered=sorted(counts.items());hist=[0]*512
  for key,count in ordered:hist[count if count<512 else 0]+=1
  emitted=[(key,count) for key,count in ordered if count>=threshold]
  expected=[z for key,count in ordered for z in (*key,count)]+[0]*((entries-len(ordered))*(ip+1))+hist+[len(emitted)]+[z for key,count in emitted for z in (*key,count)]
  expected += [len(ordered), n, len(ordered)-len(emitted)]
  # Complete output rows are sorted by original UInt32 key comparator, not physical hash slot.
  blob+=struct.pack('<8I',k,ip,n,entries,threshold,len(ordered),len(expected),len(seqs))+struct.pack('<'+'I'*len(flat),*flat)+struct.pack('<'+'I'*len(expected),*expected)
  for s in seqs:blob+=s.encode('ascii')
  cap=(256*1024)//(ip*4);bundles=(n+cap-1)//cap
  initial=entries*(ip+1)+n*ip+1
  prefix+=initial+n*ip
  # all canonical table words/hist/emitted, full no-success area+counter and guards
  outputs+=len(expected)+n*ip+1+(4+1+1+1)*16+2
  meta.append({'k':k,'ints_per_key':ip,'input_keys':n,'table_entries':entries,'unique_keys':len(ordered),'threshold':threshold,'emitted_keys':len(emitted),'source_bundle_capacity':cap,'source_add_calls':bundles,'source_stream_ids':[i%4 for i in range(bundles)],'expected_overflow_keys':0,'largest_count':max(counts.values()),'histogram_overflow_bin0':hist[0],'family_sequences':seqs})
 (P/'fixtures.bin').write_bytes(blob);(P/'fixture_manifest.json').write_text(json.dumps({'cases':meta,'fixture_bytes':len(blob),'fixture_sha256':hashlib.sha256(blob).hexdigest(),'input_prefix_count':prefix,'output_values_after_prefix':outputs,'oracle':'Independent Python DNA reverse complement/canonical basepacking and Counter; complete per-key counts,512bin count histogram and thresholded rows.','keyed_equivalence':'Every raw table row retained separately with complete invertible canonical permutation; no physical slot identity claim.'},indent=2)+'\n');print(json.dumps({'bytes':len(blob),'prefix':prefix,'outputs':outputs,'cases':[{k:v for k,v in c.items() if k!='family_sequences'} for c in meta]}))
if __name__=='__main__':main()
