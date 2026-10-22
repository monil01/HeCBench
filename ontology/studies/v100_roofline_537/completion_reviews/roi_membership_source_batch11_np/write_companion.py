import json,gzip,hashlib,re,fcntl
from pathlib import Path
O=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');h=lambda b:hashlib.sha256(b).hexdigest()
p=json.loads((O/'proposed_stage_patches.json').read_bytes());proof=json.loads(gzip.decompress((O/'proofpacket.json.gz').read_bytes()))
def evidence(rel,needles):
 path=str(W/rel);d=proof['sources'][path];b=Path(path).read_bytes();assert h(b)==d['sha256'];lines=b.decode(d['snippet_encoding']).splitlines(keepends=True);ev=[]
 for needle in needles:
  found=[i for i,l in enumerate(lines)if needle in l];assert found,(path,needle)
  for i in found:
   st=sum(len(l.encode(d['snippet_encoding']))for l in lines[:i]);raw=lines[i].encode(d['snippet_encoding']);assert b[st:st+len(raw)]==raw;ev.append({'path':path,'sha256':h(b),'line':i+1,'start_byte':st,'end_byte':st+len(raw),'text':lines[i]})
 return ev
observations=[
 {'benchmark':'ntt','observation':'The configured workload is a UInt64-to-UInt32 transform final stage, with a shared transpose and source modular word-fold arithmetic. Its certification target is exact source-policy behavior; the domain explicitly retains native carry/borrow defects and a failing independent BigInt finite-field oracle. Family inclusion must not become finite-field correctness or full-NTT certification. Three cases each time three launches plus completion; setup, warmup, input echo and guards lie outside.', 'evidence':evidence('src/ntt-julia/study_native.cu',['uint32 *__restrict__ dst','const uint64 *__restrict__ src','__shared__ uint64 buffer[512]','for(int c=0;c<3;c++){uint64*x;uint32*y;std::vector<uint64>hx']), 'scientific_intent':proof['scientific_intent']['ntt']},
 {'benchmark':'pad','observation':'The operator preserves input IEEE bits and inserts zero padding, so Float64 storage does not establish floating-point arithmetic work. Its declared timer includes resets, host-to-device suffix/flag copies, GPU launch, CPU worker start, device completion and worker join across twelve static partition paths. A kernel family is only one component; actual sampled alpha and realized overlap remain unknown.', 'evidence':evidence('src/pad-julia/study_cuda.cu',['auto step=[]','for(int shape=0;shape<2;shape++)','for(int trial=1;trial<=5;trial++){double seconds=0']), 'scientific_intent':proof['scientific_intent']['pad']},
 {'benchmark':'phmm','observation':'The source-labelled pair-HMM case is an FP64 two-state mixing/emission proxy with partial cell writes and whole physical-matrix pointer swaps. The named storage initialization and both physical outputs are part of its scope. Timed 11-by-40 update pipelines and context completion do not establish normalized probabilities, biological inference quality, or a sampled cell/phase.', 'evidence':evidence('src/phmm-julia/study_native.cu',['const int sizes[6]','void pipeline','int main(){auto f=load']), 'scientific_intent':proof['scientific_intent']['phmm']}
]
(O/'companion_scientific_observations.json').write_text(json.dumps({'scope':'Source-backed scientific distinctions only; no new numerical validation, runtime phase, whole-ROI or causal performance claim','observations':observations},indent=2)+'\n')
quotechecks=0
with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for d in proof['sources'].values():
  b=Path(d['path']).read_bytes();assert h(b)==d['sha256'];assert b[d['start_byte']:d['end_byte']]==d['snippet'].encode(d['snippet_encoding']);assert d['end_line']==b.decode(d['snippet_encoding']).count('\n')+1;quotechecks+=1
 for row in p['patches']:
  for patch in row['stage_patches']:
   d=patch['updates']['ROI_membership_proof']['exact_stage_callsite'];path=str(Path(d['path']).resolve());b=Path(path).read_bytes();assert h(b)==d['source_sha256'];lines=b.decode().splitlines();ctx=d['context'].splitlines();start=d['context_start_line']-1;assert lines[start:start+len(ctx)]==ctx,(path,d['line']);quotechecks+=1
 for d in proof['scientific_intent'].values():assert h(Path(d['binding']['path']).read_bytes())==d['binding']['sha256'];quotechecks+=1
 (O/'source_quote_revalidation.json').write_text(json.dumps({'exact_fullsource_bytes_stage_contexts_and_domain_hashes_checked':quotechecks,'all_equal':True,'metadata_lock':'LOCK_SH','source_only':True},indent=2)+'\n')
matchrows=[x for r in p['patches']for x in r['selected_observation_matching']]
counts={'models':len(p['patches']),'names':len({x['record_key']['benchmark']for x in p['patches']}),'updates':sum(len(x['stage_patches'])for x in p['patches']),'inclusions':sum(s['updates']['semantic_family_in_primary_ROI']for x in p['patches']for s in x['stage_patches']),'saved_rows':len(matchrows),'matched_saved_rows':sum(x['source_family_inclusion_only']for x in matchrows),'unassigned_saved_rows':sum(not x['source_family_inclusion_only']for x in matchrows),'inclusion_alias_nulls':sum(s['updates']['semantic_family_in_primary_ROI']and s['updates'].get('model_kernel_regex','MISSING')is None for x in p['patches']for s in x['stage_patches'])}
(O/'review_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts)
