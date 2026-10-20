"""Full CPU-only input-bit/trace/guard/field/hash audit; no GPU executions."""
from pathlib import Path
import argparse,array,struct,json,hashlib,math
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=Path(__file__).resolve().parent;R=B.parents[1];S=R/'.porting-state/portable_julia';data=(S/'external_assets/gmm/data').read_bytes();assert len(data)==33000
rows=[list(map(float,line.split()))for line in data.decode().splitlines()if line.strip()];assert len(rows)==1000 and all(len(v)==2 for v in rows);points=rows[1:993];raw=[v for row in points for v in row];transpose=[row[d]for d in range(2)for row in points];bits=lambda xs:[struct.unpack('<I',struct.pack('<f',v))[0]for v in xs];guard=0xa5a5a5a5;gval=struct.unpack('<f',struct.pack('<I',guard))[0]
sizes=[6,6,6,6,12,24,24,11904];prefix=array.array('d',[2,992,6])
for i,words in list(enumerate([[0]*n for n in sizes]))+[(9,bits(raw)),(10,bits(transpose))]:
 w=[guard]*16+words+[guard]*16;prefix.extend([i,len(w)]);prefix.extend(w)
assert len(prefix)==16299
x=json.loads(a.result.read_text());assert x['matched_performance_valid'] and all(x['variant_verified'].values())
first=Path(x['models']['cuda']['trials'][0]['output_artifact']['saved_path']).read_bytes();v=array.array('d');v.frombytes(first);assert v[:len(prefix)]==prefix;k=len(prefix);stages=[];selected_snapshots={};checks=0
for step in range(278):
 header=list(map(int,v[k:k+7]));assert all(float(q)==vv for q,vv in zip(header,v[k:k+7]));stage,gx,gy,threads,d,m,n=header;k+=7
 assert 1<=stage<=7 and 1<=m<=6 and d==2 and n==992 and threads==256
 assert gx==(1 if stage==1 else 24 if stage==4 else math.ceil(m/6)if stage==7 else m)
 assert gy==(24 if stage==3 else 2 if stage==6 else 3 if stage==7 else 1)
 def snapshot(k):
  values=[]
  for size in sizes:
   assert v[k]==size+32;k+=1;q=v[k:k+size+32];k+=size+32;assert all(z==gval for z in q[:16])and all(z==gval for z in q[-16:]);assert all(math.isfinite(z)for z in q);values.append(q[16:-16])
  like=int(v[k]);k+=1
  if stage==4:assert like==56;q=v[k:k+like];k+=like;assert all(z==gval for z in q[:16])and all(z==gval for z in q[-16:])
  else:assert like==0
  return k,values
 k,before=snapshot(k);k,after=snapshot(k);stages.append(header)
 if stage==4:selected_snapshots[m]=after
# Verify the full adaptive pipeline grammar and allsource modelorders.
i=2;assert [q[0]for q in stages[:2]]==[1,2];iterations=[]
for m in range(6,0,-1):
 assert [q[0]for q in stages[i:i+2]]==[3,4]and all(q[5]==m for q in stages[i:i+2]);i+=2;it=0
 while i<len(stages)and stages[i][0]==5:
  assert [q[0]for q in stages[i:i+6]]==[5,6,7,2,3,4]and all(q[5]==m for q in stages[i:i+6]);i+=6;it+=1
 assert 1<=it<=200;iterations.append({'clusters':m,'EM_updates':it})
assert i==278 and sum(q['EM_updates']for q in iterations)==44
assert v[k]==3 and v[k+1]==278;k+=2
for values,count in zip(selected_snapshots[3],[3,3,3,3,6,12,12,2976]):
 assert v[k:k+count]==values[:count];k+=count
assert k==len(v)==6839533
records=[];expected_hash=hashlib.sha256(first).hexdigest()
for model,m in x['models'].items():
 assert m['complete']and m['timing_valid']and m['provenance_valid']and len(m['trials'])==5
 times=[q['time_seconds']for q in m['trials']];assert math.isclose(sum(times[1:])/4,m['mean_seconds'],rel_tol=1e-15)
 for trial in m['trials']:
  q=trial['output_artifact'];f=Path(q['saved_path']);b=f.read_bytes();h=hashlib.sha256(b).hexdigest();assert h==q['sha256']==expected_hash and b==first;assert trial['correct']and trial['checked_values']==508138 and trial['active_final_clusters']==3 and trial['selected_GPU_stages']==278
  records.append({'model':model,'trial':trial['trial'],'path':str(f),'bytes':len(b),'sha256':h,'full_exact_input_prefix':16299,'complete_guarded_trace_values':len(v),'actual_independent_stage_checks':508138,'full_raw_identity_with_native_capture':True})
review={'all_pass':True,'result_path':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'full_artifact_hash_checks':records,'artifacts':20,'actual_input_prefix_values':16299,'full_values_each':6839533,'all7entrypoints_and_full_source_EM_modelorder_pipeline':True,'stage_count':278,'EM_updates_by_modelorder':iterations,'selected_cluster_count':3,'guard_checks_all_prepoststates':True,'source_saved_best_model_exactly_fromcaptured_GPU_fields':True,'actual_data_source_sha256':hashlib.sha256(data).hexdigest(),'statistical_scope':'IndependentGaussian/responsibility/moment/analyticinverse checks for every actualstage, sourcelegacymerge/base10logdet retained; no globalnonconvexfit or statisticalgroundtruth claim'};t=a.output.with_suffix('.tmp');t.write_text(json.dumps(review,indent=2)+'\n');t.replace(a.output);print(json.dumps({'all_pass':True,'artifacts':20,'values_each':len(v),'EM_updates':iterations}))
