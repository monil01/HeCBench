#!/usr/bin/env python3
from pathlib import Path
import argparse,array,hashlib,json
ap=argparse.ArgumentParser();ap.add_argument('result');ap.add_argument('--output',required=True);a=ap.parse_args();rpath=Path(a.result);r=json.loads(rpath.read_text());t=r['models']['cuda']['trials'][0];artifact=Path(t['output_artifact']['saved_path']);v=array.array('I');v.frombytes(artifact.read_bytes());at=0;first={};reports=[]
while at<len(v):
 seg,size,threads,trial,n=v[at:at+5];at+=5;initialized=v[at:at+size+128];at+=size+128;final=v[at:at+size+128];at+=size+128;features=v[at:at+n+256];at+=n+256
 assert size==(seg//256+1)*256 and threads==((n+255)//256)*256 and n==seg*128
 assert all(initialized[64+i]==(26757677*(5551212^i)+1)%(1<<31)for i in range(size))
 for state in [initialized,final]:assert all(x==0xa5a5a5a5 for x in state[:64]+state[64+size:])
 assert all(x==0xa5a5 for x in features[:128]+features[128+n:]);payload=features[128:128+n];assert all(x<=65535 for x in payload)
 if trial==0:first[seg]=payload
 differences=[i for i,(x,y)in enumerate(zip(payload,first[seg]))if x!=y];reports.append({'seg':seg,'trial':trial,'state_size':size,'threads':threads,'half_features':n,'different_half_bits_vs_repeat0':len(differences),'first_difference':{'index':differences[0],'repeat0_bits':first[seg][differences[0]],'actual_bits':payload[differences[0]]}if differences else None,'actual_initialized_state_sha256':hashlib.sha256(initialized.tobytes()).hexdigest(),'actual_feature_bits_encoded_u32_sha256':hashlib.sha256(payload.tobytes()).hexdigest(),'guards_verified':True})
assert at==len(v) and len(reports)==15
out={'pass':True,'scope':'Actualunchangednativefeatureinitializerrepeats, notport/correctness/performanceacceptance. EveryinitializedRNGstateexact, allphysicalguardschecked, completehalfbitdifferencescomputedindependently.','result':str(rpath),'result_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),'artifact':str(artifact),'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'reports':reports,'observed_initial_feature_nondeterminism':any(x['different_half_bits_vs_repeat0']for x in reports),'time_seconds_field_semantics':'Rawdiagnosticrow0 is an explicitly unmeasured sentinel, notactualruntime. Incompleteone-row/JITfalse/diagnostic_excluded prohibitsperformance/primaryinclusion; no benchmarktimeclaim.'};Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print({'pass':True,'variation':out['observed_initial_feature_nondeterminism'],'max_half_differences':max(x['different_half_bits_vs_repeat0']for x in reports)})
