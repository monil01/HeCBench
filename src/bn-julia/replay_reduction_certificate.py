"""Recompute original score/index reduction on an actual preserved BN case.
Candidate score vector is independently reconstructed from captured local-score
array and actual parent mask, not an unrecorded GPU internal-state recovery.
"""
from pathlib import Path
import array,itertools,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1]
a=array.array('d');path=R/'.porting-state/porting_logs/bn-julia-shared/native_preparation_canonical_trials/output_trial1.bin';a.frombytes(path.read_bytes());node=5;step=3;mask=a[7872+(step*9+node)*73+32:7872+(step*9+node)*73+41];available=[i for i,b in enumerate(mask)if b];parents=[c for n in range(5)for c in itertools.combinations(available,n)];globalcomb=[c for n in range(5)for c in itertools.combinations([i for i in range(9)if i!=node],n)];rank={p:i for i,p in enumerate(globalcomb)};local=a[220714:220714+1467];scores=[local[node*163+rank[p]]for p in parents];sentinel=-999999986991104.0

def replay(tiefix):
 shared=scores+[sentinel]*(256-len(scores))
 for stride in (128,64,32,16,8,4,2,1):
  for tid in range(stride):
   left,right=shared[tid],shared[tid+stride]
   if right>left and right<0:shared[tid]=right;shared[tid+stride]=float(tid+stride)
   elif (right<=left if tiefix else right<left)and left<0:shared[tid+stride]=float(tid)
   elif left>0 and right<0:shared[tid]=right;shared[tid+stride]=float(tid+stride)
   elif left<0 and right>0:shared[tid+stride]=float(tid)
 t=0;trace=[]
 for i in range(7):
  if not 0<=t<128:break
  address=2**i+t;t=int(shared[address]);trace.append({'address':address,'next':t})
 return {'score':shared[0],'winner_thread':t,'winner_parents':parents[t]if 0<=t<len(parents)else None,'winner_parent_score':scores[t]if 0<=t<len(scores)else None,'trace':trace}
original=replay(False);tieonly=replay(True);best=max(scores);expected=scores.index(best)
raw=a[220714+1467+step*618+node*5:220714+1467+step*618+node*5+5]
out={'source_artifact':str(path.relative_to(R)),'source_artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'case':0,'iteration':step,'node':node,'actual_parent_mask':list(mask),'reconstructed_candidate_parents':parents,'reconstructed_candidate_scores':scores,'captured_score_and_encoded_parents':list(raw),'original_reduction_replay':original,'tieonly_reduction_replay':tieonly,'independent_argmax':{'score':best,'thread':expected,'parents':parents[expected]},'interpretation':'Inputcandidatevector independentlyreconstructedfrom retainedactualmask/localtable; notclaimofrecoveringmissingGPUinternaltrace. Currenttieonlynativecandidate returnscorrectscorebutwrongparentset. Destructivescore/indexstorage doesnotpreserveargmaxcertificate.'}
assert tieonly['score']==raw[0]==best;assert tuple(tieonly['winner_parents'])==(3,);assert tieonly['winner_parent_score']!=best
(P/'reduction_certificate_witness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if not k.startswith('reconstructed')},indent=2))
