from pathlib import Path
import json,struct,shutil,hashlib
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent
# Explicit full source geometries, named varied exact-dyadic feature content.
CASES=[(1000,0,2000,10,11,0,1234),(0,1000,2000,10,11,0,1234),(1000,1000,2000,10,11,0,1234),(3,6,17,3,3,0,1234),(0,4,1,2,1,0,1234),(0,2,17,2,17,0,9876),(5,4,257,2,11,0,1234)]
# uint64wrap+mod2^63 independent integer generator; Float64conversion/multiply mirrors actual declared index arithmetic.
def stream_indices(seed,cols,k):
 z=seed;selected=set();draws=[]
 while len(selected)<k:
  z=(2806196910506780709*z+1)&((1<<63)-1);index=int(float(z)/(1<<63)*cols);assert 0<=index<cols;selected.add(index);draws.append(index);assert len(draws)<100000
 return sorted(selected),draws
allmasks=bytearray();features=[];descs=[];historical=[]
for exact,sampled,cols,bg,kmax,mode,seed in CASES:
 rows=exact+sampled;mo=len(allmasks);fo=len(features);ns=[kmax-i%2 for i in range(sampled//2)];assert all(0<=k<=min(cols,256)for k in ns)
 mask=bytearray(rows*cols)
 for row in range(exact):mask[row*cols+row]=1;mask[row*cols+row+1]=1
 draws={}
 for pair,k in enumerate(ns):
  chosen,trace=stream_indices(seed,cols,k);draws[str(k)]=trace;chosen=set(chosen)
  for col in range(cols):mask[(exact+2*pair)*cols+col]=int(col in chosen);mask[(exact+2*pair+1)*cols+col]=int(col not in chosen)
  if pair==1:
   old,_=stream_indices(seed+pair,cols,k);historical.append(dict(cols=cols,k=k,block=pair,source_same_seed_indices=sorted(chosen),historical_seed_plus_block_indices=old,masks_differ=set(old)!=chosen))
 # Dyadic fixtures vary by donor row AND feature; observation distinct from every donor.
 background=[-32.+row/4+col/2048 for row in range(bg)for col in range(cols)];observation=[128.+col/32 for col in range(cols)];features.extend(background+observation);allmasks+=mask;descs.append(dict(exact=exact,sampled=sampled,cols=cols,bg=bg,kmax=kmax,seed=seed,n=rows*cols,dataset_n=rows*bg*cols,mask_offset=mo,feature_offset=fo,nsamples=ns,draw_traces_by_k=draws))
(B/'oracle_masks_u8.bin').write_bytes(allmasks);(B/'features_f32.bin').write_bytes(struct.pack('<'+'f'*len(features),*features));(B/'cases.json').write_text(json.dumps(dict(cases=descs,historical_seed_policy_witness=historical,proof='Eachactive lane startsidenticalseed andadvancesonlyonoccupiedatomicexchange. Claimedset isfirstkdistinctLCGindices regardlesswinnerinterleaving: toclaimlaterdistinctentryallprioronesmustalreadybeoccupied. Keeporiginalkconcurrentlanes/Float32atomics, neverreplacewithserialsamplertomatch.',scope='KernelSHAPcoalitiondatasetgeneration, no model evaluation/Shapleyvalueestimation/independentcoalitionqualitycertificate. All3sourcefullgeometries, commonvarieddyadiccontentsnotoriginalconstantbackground/overflowingsentinelinitializer. Samebyvalueseedperblock retained, repeatedcoalitionsacrossblocks observedbysourceproof.'),indent=2)+'\n')
(B/'study_cases.h').write_text('struct Case{int exact,sampled,cols,bg,kmax,mo,fo;unsigned long long seed;};\nstatic Case cases[]={'+','.join('{'+','.join(str(c[k])for k in ['exact','sampled','cols','bg','kmax','mask_offset','feature_offset'])+','+str(c['seed'])+'ULL}'for c in descs)+'};\n');(B/'study_cases.jl').write_text('const CASES='+repr([(c['exact'],c['sampled'],c['cols'],c['bg'],c['kmax'],c['mask_offset'],c['feature_offset'],c['seed'])for c in descs])+'\n')
prefix=sum(c['n']+(c['bg']+1)*c['cols']+c['sampled']//2+5*32+4 for c in descs);post=sum(c['n']+c['dataset_n']+(c['bg']+1)*c['cols']+c['sampled']//2+5*32 for c in descs);print('masks',len(allmasks),'prefix',prefix,'artifact words',prefix+post)
