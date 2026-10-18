"""CPU-only saved output/independent edit-distance audit, no new GPU work."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;F=Path(sys.argv[1]).resolve();r=json.loads(F.read_text());N=1025
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=np.fromfile(P/'study_reads_u32.bin',dtype='<u4');b=np.fromfile(P/'study_refs_u32.bin',dtype='<u4');expected=np.fromfile(P/'oracle_results_i32.bin',dtype='<i4').reshape(26,N);dist=np.fromfile(P/'exact_edit_distance_i32.bin',dtype='<i4');seqs=[line.split('\t')for line in(P/'study_sequences.tsv').read_text().splitlines()]
assert len(seqs)==N
# A second independent, ordinary dynamic-programming recurrence checks all exact distances.
def dp(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
for i,(x,y)in enumerate(seqs):
 assert dp(x,y)==dist[i]
 for packed,s in [(a,x),(b,y)]:
  decoded=''.join('ACGT'[(int(packed[i*8+j//16])>>(30-2*(j%16)))&3]for j in range(100));assert decoded==s
prefix=np.r_[a,b,np.array([N,100,25,26,20,256,8,0],dtype='<u4')];assert len(prefix)==16408
rows=[];bindings={};baseline=None
for model,v in r['models'].items():
 if v['status']=='missing':continue
 assert v['status']=='native_check_pass'and len(v['trials'])==5
 for path,h in v['adapter_file_hashes'].items():assert sha(path)==h;bindings[path]=h
 for t in v['trials']:
  art=t['output_artifact'];assert sha(art['saved_path'])==art['sha256'];raw=np.fromfile(art['saved_path'],dtype='<u4');assert len(raw)==43506 and raw[:len(prefix)].tobytes()==prefix.tobytes();out=raw[len(prefix):len(prefix)+26*N].reshape(26,N);assert np.array_equal(out,expected)and np.all(raw[-448:]==0x5a5a5a5a)
  if baseline is None:baseline=art['sha256']
  assert baseline==art['sha256'];quality=[]
  for T in range(26):
   accepted=out[T]==1;close=dist<=T;quality.append(dict(threshold=T,accepted=int(accepted.sum()),false_accepts=int(np.sum(accepted&~close)),false_rejects=int(np.sum(~accepted&close))))
  rows.append(dict(model=model,trial=t['trial'],artifact_sha256=art['sha256'],artifact_count=len(raw),exact_input_bytes=4*len(prefix),full_outputs=26*N,guards=448,quality_vs_exact_distance=quality))
proof=dict(result_path=str(F.relative_to(R)),result_sha256=sha(F),NumPy=np.__version__,method='Rehash every source/fixture/capture; exact input prefix/full26binaryvectors/448guards; independently decode everypackedsequence and ordinary exactLevenshteinDP for all1025pairs, confirm separatelygeneratedbitvectorDPdistances. Filterverdict fidelity isseparatefromscientificclassificationquality.',rows=rows,executed_source_bindings=bindings)
(P/'saved_output_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
witnesses=[]
for label,mask in [('false_reject',(expected==0)&(dist[None,:]<=np.arange(26)[:,None])),('false_accept',(expected==1)&(dist[None,:]>np.arange(26)[:,None]))]:
 indices=np.argwhere(mask)
 if len(indices):
  T,i=map(int,indices[0]);witnesses.append(dict(kind=label,threshold=T,pair_index_zero_based=i,read=seqs[i][0],reference=seqs[i][1],exact_Levenshtein_distance=int(dist[i]),native_filter_verdict=int(expected[T,i]),output_artifact_word_zero_based=len(prefix)+T*N+i,total_observed_pair_threshold_cases=int(len(indices))))
q=dict(classification='known_native_scientific_filter_defect'if any(x['kind']=='false_reject'for x in witnesses)else'filter_quality_characterization',unaltered_native_source=True,source_kernel={'path':'src/snake-cuda/kernel.h','sha256':sha(R/'src/snake-cuda/kernel.h')},source_main={'path':'src/snake-cuda/main.cu','sha256':sha(R/'src/snake-cuda/main.cu')},actual_native_artifact=r['models']['cuda']['trials'][0]['output_artifact'],source_operational_CPU_agreement=True,independent_exact_distances='TwoindependentCPUalgorithms:bitvectorDP andordinarydynamicprogramming all1025pairs',witnesses=witnesses,scope='100baseDNA andthresholds0..25; falseacceptsareexpectedpossibleforaprealignmentfilter, falserejectsviolateanexact-edit-distance-safe-prefilter interpretation. Noactualalignment orbiologicalvalidation claim. Rootcauseofsourcefilterfalse rejects notisolated; customwordshiftidentityquirks retained.',all_models_inherit_native_verdicts=True,all_trial_quality=rows[0]['quality_vs_exact_distance'])
witness_output=P/('filter_quality_final.json'if(P/'native_quality_witness.json').exists()else'native_quality_witness.json');witness_output.write_text(json.dumps(q,indent=2)+'\n');print('AUDITED',len(rows),'captures',len(bindings),'bindings; quality',[(x['kind'],x['total_observed_pair_threshold_cases'])for x in witnesses])
