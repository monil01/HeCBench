from pathlib import Path
import struct,re,json,hashlib
base=Path('/tmp/columnarSolver_native');cpu=base/'cpu_separate.bin';actualcpu=Path('/tmp/maps_columnar_lastcandidate_full_review.bin');assert actualcpu.read_bytes()==cpu.read_bytes()
rows={}
for line in Path('/tmp/maps_columnar_lastcandidate_full_review.log').read_text().splitlines():
 t,b,last,acc=map(int,re.match(r'REVIEW row=(\d+) best0=(\d+) last0=(\d+) lastAccept=(\d+)',line).groups());assert t not in rows;rows[t]={'best0':b,'last_candidate0':last,'last_accepted_cycle':acc}
assert len(rows)==1024
cv=struct.unpack('<472064I',cpu.read_bytes());results={}
for name,folder in [('nvcc12.0','diagnostic01'),('nvcc12.9','diagnostic_nvcc129')]:
 f=base/folder/'trial0_state.bin';gv=struct.unpack('<472064I',f.read_bytes());mismatch=[];lastmismatch=[];other=[]
 for t,x in rows.items():
  start=1024+t*397;assert cv[start]==x['best0']
  if gv[start]!=x['best0']:mismatch.append(t)
  if gv[start]!=x['last_candidate0']:lastmismatch.append(t)
  if gv[start+1:start+397]!=cv[start+1:start+397]:other.append(t)
 assert gv[:1024]==cv[:1024] and gv[1024+1024*397:]==cv[1024+1024*397:]
 results[name]={'raw_gpu_capture':str(f),'raw_gpu_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'raw_gpu_bytes':f.stat().st_size,'first_character_differs_from_accepted_best_rows':len(mismatch),'first_character_differs_from_last_candidate_rows':len(lastmismatch),'other396characters_differ_rows':len(other),'fullseed_and_63diagwords_exact':True,'first_character_wrong_rows':mismatch,'counterexamples_to_last_candidate_pattern':lastmismatch}
assert results['nvcc12.0']['raw_gpu_sha256']==results['nvcc12.9']['raw_gpu_sha256']
files=[Path('/tmp/maps_columnar_lastcandidate_full_review.cpp'),Path('/tmp/maps_columnar_lastcandidate_full_review.log'),actualcpu,Path('/tmp/maps_columnar_compare_lastcandidate.py'),cpu]
report={'scope':'Read-only independentCPUreview; fixedsuppliedacademicfixture only. Originalsource/sharedstate/Git untouched, noGPUcommands orOSchanges. Owner retains allacceptance checks.','full_trajectories':1024,'cycles_per_trajectory':150000,'CPU_instrumentation':'CPU-sourcecopyonlyadds lastacceptedcycle+bestLocal[0]/localDecrypted[0] prints andcompleteCPUrawoutput. Original sourcecpu finalbytes exactlysame, nochange ofmath/RNG/draws/checker.','CPU_review_raw_equals_prior_full_CPU_oracle':True,'observations':results,'hypothesis_not_proven':'GPU firstsavedbest character followslastcandidate lifetime; suggests compiler/storage-lifetime issue, but patternalone doesnotestablishcause. Doesnotjustifyrelaxing key→plaintext/score consistency orcallingnativealgorithm universallywrong.','narrow_control_proposal':'Owner comparesvolatilebestLocalDecrypted storage orseparateexplicitfirstscalar bestvalue update, preservingall397 bestvalues/condition andfull150000schedule. FullrawCPU+independentkey/score checks remainstrict; archive source qualifier/control separately, noacceptancerevision.','files':[{'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in files],'perrow_CPU_trace':rows}
out=Path('/tmp/maps_columnar_lastcandidate_full_review.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'models':{m:{k:v for k,v in r.items()if k not in ['first_character_wrong_rows','counterexamples_to_last_candidate_pattern']}for m,r in results.items()}}))
