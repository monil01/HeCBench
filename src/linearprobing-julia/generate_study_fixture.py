from pathlib import Path
import json
p=Path(__file__).resolve().parent
MOD=1<<32;CAP=64*1024*1024;EMPTY=MOD-1

def mix(k):
 k^=k>>16;k=k*0x85ebca6b%MOD;k^=k>>13;k=k*0xc2b2ae35%MOD;k^=k>>16;return k

def unmix(k):
 k^=k>>16;k=k*pow(0xc2b2ae35,-1,MOD)%MOD;k=k^(k>>13)^(k>>26);k=k*pow(0x85ebca6b,-1,MOD)%MOD;k^=k>>16;return k
keys=[]
for bucket in [0,CAP-1,917,8191]:
 for j in range(32):
  k=unmix(bucket+j*CAP);assert mix(k)&(CAP-1)==bucket and k!=EMPTY;keys.append(k)
i=1
while len(keys)<1027:
 k=(i*104729+17)%EMPTY;i+=1
 if k not in keys:keys.append(k)
assert keys[0]==0 and len(set(keys))==1027
vals={k:(mix(k^0xa5a5a5a5)%EMPTY) for k in keys};vals[0]=123
batch1=[(0,123)];bulk=[(k,vals[k]) for k in keys[1:]]+[(k,vals[k]) for k in keys[32:64]]
updates=[(k,(vals[k]+777)%EMPTY) for k in [0]+keys[128:160]]
deleted=keys[1::3];absent=[unmix(0+63*CAP),unmix(CAP-1+63*CAP)];assert not any(k in vals for k in absent)
dels=[(k,0) for k in deleted+absent];reinserts=[(k,(vals[k]+999)%EMPTY) for k in deleted[:16]]
batches=[batch1,bulk,updates,dels,reinserts];counts=[len(a) for a in batches];assert counts==[1,1058,33,344,16]
# Two language fixtures contain identical source-defined UInt32 arrays, generated once.
lines=['# Deterministic numeric dictionary fixture; source generator is generate_study_fixture.py.','const STUDY_BATCHES = (']
for a in batches:lines.append(' (UInt32['+','.join(str(k) for k,v in a)+'],UInt32['+','.join(str(v) for k,v in a)+']),')
lines+=[')',''];(p/'study_fixture.jl').write_text('\n'.join(lines))
h=['// Same numeric dictionary fixture as study_fixture.jl.','#pragma once','struct StudyBatch { std::vector<KeyValue> pairs; };','std::vector<StudyBatch> study_batches(){return {']
for a in batches:h.append('StudyBatch{{'+','.join('{'+str(k)+'u,'+str(v)+'u}' for k,v in a)+'}},')
h+=['};}',''];(p/'study_fixture.h').write_text('\n'.join(h))
fixture={'capacity':CAP,'empty':EMPTY,'unique_keys':1027,'batch_sizes':counts,'collision_groups':[{'bucket':x,'keys':32} for x in [0,CAP-1,917,8191]],'expected_snapshot_sizes':[1027,685,701],'semantics':'Identical-valued duplicates within bulk batch, distinct updates in later ordered batch; unique deletes incl2absent,16same-keytombstone reinserts. Key0 retained inbucket0, valueupdated.','key_value_input_prefix_values':2914,'canonical_public_values':4829,'artifact_values':7743};(p/'fixture_definition.json').write_text(json.dumps(fixture,indent=2)+'\n')
