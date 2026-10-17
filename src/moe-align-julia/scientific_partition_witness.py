"""Independent exact token multiset/partition/padding witness for saved captures."""
from pathlib import Path
import array,collections,json,sys
rp=Path(sys.argv[1]).resolve();res=json.loads(rp.read_text());shapes=[(3,32,3,False),(127,64,4,False),(3,128,3,False),(600,32,3,False),(257,128,4,True)];prefix=[];expected=[];cases=[]
for tokens,E,K,mapped in shapes:
 N=tokens*K;ids=[E+2 if mapped and i%17==0 else(i//K*17+i%K*13)%E for i in range(N)];mapping=[-1 if mapped and e%7==0 else E-1-e if mapped else e for e in range(E)];maxpad=(N+E*31+31)//32*32
 if N<E:maxpad=min(N*32,maxpad)
 groups={e:[]for e in range(E)};excluded=[]
 for i,e in enumerate(ids):
  if e>=E:excluded.append(i);continue
  target=mapping[e]if mapped else e
  if target<0:excluded.append(i);continue
  groups[target].append(i)
 prefix.extend(ids+mapping+[tokens,E,K,int(mapped),N,maxpad,maxpad//32,3]);sorted_ids=[];expert_ids=[];cum=[0];counts=[]
 for e,group in groups.items():
  assert group==sorted(group);padded=(len(group)+31)//32*32;sorted_ids.extend(group+[N]*(padded-len(group)));expert_ids.extend([e]*(padded//32));cum.append(cum[-1]+padded);counts.append(len(group))
 total=len(sorted_ids);sorted_ids.extend([N]*(maxpad-total));expert_ids.extend([0]*(maxpad//32-len(expert_ids)));small=N<1024 and E<=64;output=sorted_ids+expert_ids+[total]+([]if small else[cum[e]+counts[e]for e in range(E)]+[total]);expected.extend(output);valid=N-len(excluded)
 cases.append({'tokens':tokens,'experts':E,'topk':K,'small':small,'mapped_invalid':mapped,'valid_assignments':valid,'excluded_assignments':len(excluded),'padded_total':total,'output_capacity':maxpad,'padding_inside_active_blocks':total-valid,'inactive_capacity_slots':maxpad-total,'padding_amplification':total/valid,'output_words':len(output),'counts':counts})
report={'oracle':'Independent exact grouping/multiset preservation with per-expert ceil-to32 padding; all expert labels and final atomic-prefix endpoints independently derived.','input_prefix_count':len(prefix),'output_count':len(expected),'cases':cases,'trials':[],'evidence_strength':'Canonical complete semantic outputs, within-expert order intentionally removed because native atomic scatter ordering is nondeterministic.','scope':'Fivecases, singlemodeloffset0 and null tokenmask. Small path invalid>=E IDs not tested. No downstream expert-GEMM throughput or model quality claim.'}
for model,spec in res['models'].items():
 for t in spec.get('trials',[]):
  data=array.array('i');data.frombytes(Path(t['output_artifact']['saved_path']).read_bytes())
  if sys.byteorder!='little':data.byteswap()
  assert list(data[:3778])==prefix,(model,'inputs');out=list(data[3778:]);assert out==expected,(model,'full partition/padding/prefix')
  report['trials'].append({'model':model,'trial':t['trial'],'exact_prefix':True,'all_canonical_tokens_labels_padding_total_prefix_exact':True})
q=rp.parent/'scientific_partition_witness.json';q.write_text(json.dumps(report,indent=2)+'\n');print(q)
