"""Full saved-state/modular oracle audit; hashes deduplicate identical captures."""
from pathlib import Path
import json,struct,hashlib,math
from generate_fixture import source,initial,expected,GUARD
B=Path(__file__).resolve().parent;R=B.parents[1];p=R/'.porting-state/portable_julia/runs/20261018T044443543094Z_interleave/result.json';r=json.loads(p.read_text());P=4096;F=16;G=16;W=P*F+2*G;cache={};rows=[];issues=[];timings=[]
def check_payload(data):
 vals=struct.unpack('<'+'I'*(len(data)//4),data);assert len(vals)==24*W;path=0;bad=0
 for fixture in range(2):
  for n in [4096,4093]:
   for layout in range(2):
    src=vals[path*2*W:(path*2+1)*W];dst=vals[(path*2+1)*W:(path*2+2)*W];out=vals[(16+path)*W:(17+path)*W]
    for v in [src,dst,out]:bad+=v[:G]!=(GUARD,)*G or v[-G:]!=(GUARD,)*G
    for i in range(P):
     for f in range(F):
      index=G+(i*F+f if layout==0 else f*P+i);bad+=src[index]!=source(i,f,fixture);bad+=dst[index]!=initial(i,f,fixture);bad+=out[index]!=expected(i,f,fixture,n)
    path+=1
 return {'bad_values_or_guard_groups':bad,'pass':bad==0,'values':len(vals)}
for model,m in r['models'].items():
 for trial in m['trials']:
  a=trial['output_artifact'];data=Path(a['saved_path']).read_bytes();sha=hashlib.sha256(data).hexdigest();assert sha==a['sha256']
  if sha not in cache:cache[sha]=check_payload(data)
  row={'model':model,'trial':trial['trial'],'artifact_sha256':sha,**cache[sha]};rows.append(row)
  if not row['pass']:issues.append(row)
  assert len(trial['path_times_seconds'])==8 and math.isclose(sum(trial['path_times_seconds']),trial['time_seconds'],rel_tol=1e-9)
 for path in range(8):
  fixture=path//4;shape=(path%4)//2;layout=path%2;ts=[t['path_times_seconds'][path] for t in m['trials']];timings.append({'model':model,'path':path,'fixture':fixture,'active_records':4096 if shape==0 else 4093,'layout':'AoS' if layout==0 else 'SoA','trial_times_seconds':ts,'mean_last_four_seconds':sum(ts[1:])/4,'retained_min_seconds':min(ts[1:]),'retained_max_seconds':max(ts[1:])})
original=(R/'src/interleave-cuda/main.cu').read_text();segment=original[original.index('__global__ void add_kernel_interleaved'):original.index('void add_test_interleaved(')].strip();copied=(B/'native_kernels.inc').read_text();copied=copied[copied.index('__global__ void add_kernel_interleaved'):].strip();assert copied==segment
summary={'result':str(p.relative_to(R)),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checked_artifacts':len(rows),'unique_payloads_checked':len(cache),'complete_values_per_artifact':24*W,'input_prefix_values_per_artifact':16*W,'all16fields_and_all4096physicalrecords_including_inactive_tail':True,'independent_oracle':'Arbitrary-precision initial+4096*source modulo2^32, physical AoS/SoA indexing; allguards andreadonlybuffers exact','native_kernel_bodies_verbatim':True,'native_kernel_body_sha256':hashlib.sha256(segment.encode()).hexdigest(),'checks':rows,'issues':issues}
(B/'output_audit.json').write_text(json.dumps(summary,indent=2)+'\n');(B/'path_timing_summary.json').write_text(json.dumps({'result':str(p.relative_to(R)),'configuration_id':r['configuration_id'],'rows':timings,'mean_protocol':'Arithmetic mean trials2-5; all5correct; rangesare retainedmin/max, notCI','timing_scope':'Sum8separate kernel launch/contextcompletion windows, reset/copy/setup/readback outside each','limits':'No dynamic traffic/instruction/occupancy profile; original16UInt32fields/COUNT4096 fullcapacity andtail namedonly'},indent=2)+'\n');print(json.dumps({'artifacts':len(rows),'unique_payloads':len(cache),'issues':issues}));assert len(rows)==20 and not issues
