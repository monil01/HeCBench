#!/usr/bin/env python3
"""Independent complete physical/primary/operator/model joins, CPU-only."""
from pathlib import Path
import argparse,array,hashlib,json,math,struct,subprocess
ap=argparse.ArgumentParser();ap.add_argument('result');ap.add_argument('--model',required=True);ap.add_argument('--process-id',type=int,required=True);ap.add_argument('--raw-root',required=True);ap.add_argument('--operator-version',type=int,choices=[1,2],default=2);ap.add_argument('--model-oracle',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=Path(__file__).resolve().parent;rpath=Path(a.result);r=json.loads(rpath.read_text());fixture=(p/'data/mf_sgd_source_inputs.bin').read_bytes();at=0
magic,version,m,n,k,nrates,nnz,nt=struct.unpack_from('<2I4i2q',fixture);at=40;parameters=struct.unpack_from('<9f',fixture,at);at+=36

def arr(T,b):
 x=array.array(T);x.frombytes(b);return x
maps=arr('i',fixture[at:at+8*(m+n)]);at+=8*(m+n);R=fixture[at:at+12*nnz];at+=12*nnz;test=fixture[at:at+12*nt];at+=12*nt;initialP=fixture[at:at+2*m*k];at+=2*m*k;initialQ=fixture[at:at+2*n*k];at+=2*n*k;rates=fixture[at:at+nrates*4];at+=nrates*4;assert at==len(fixture);original=(p/'data/movielens_u1_train.bin').read_bytes();f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];source_scale=f32(math.sqrt(1./128));lcg=lambda s:(26757677*s+1)%(1<<31);H=lambda v:struct.pack('<e',v)
unitP=b''.join(H(((i*17+j*7)%127-63)/(1024*(1<<((i+j*3)%13 if a.operator_version==2 else 0))))for i in range(257)for j in range(128));unitQ=b''.join(H(((i*11+j*3)%113-56)/(1024*(1<<((i*5+j)%11 if a.operator_version==2 else 0))))for i in range(257)for j in range(128));unitR=b''.join(struct.pack('<iif',i,i,(i%7-3)*.25)for i in range(257));oracle=(p/('operator_oracle_v2.bin'if a.operator_version==2 else'operator_oracle.bin')).read_bytes();assert struct.unpack_from('<5I',oracle)==(0x4d464f31,257,128,12,3072);selected=struct.unpack_from('<12i',oracle,20);unit_expect=list(struct.iter_unpack('<3idd',oracle[68:]));assert len(unit_expect)==3072
common=array.array('d',[magic,version,m,n,k,nnz,nt,*parameters]);common.extend(maps)
for node in struct.iter_unpack('<iif',original):common.extend(node)
for node in struct.iter_unpack('<iif',test):common.extend(node)
common.extend((257,128,12,f32(.0125),f32(.05),f32(.075)));assert len(common)==305272

def raw_case(base):
 raw=Path(str(base)+'.raw.bin').read_bytes();guards=Path(str(base)+'.guards.u8').read_bytes();rows=[json.loads(x)for x in Path(str(base)+'.layout.jsonl').read_text().splitlines()];assert guards and set(guards)=={165};offset=go=0;payloads={}
 for row in rows:
  N=row['payload_bytes'];assert row['raw_offset_bytes']==offset and row['guard_offset_bytes']==go;phys=raw[offset:offset+512+N];assert len(phys)==512+N;actual=phys[:256]+phys[-256:];assert actual==guards[go:go+512];key=(row['event'],row['id']);assert key not in payloads;payloads[key]=phys[256:-256];offset+=N+512;go+=512
 assert offset==len(raw)and go==len(guards);return payloads,{'raw_path':str(base)+'.raw.bin','raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'actual_guard_path':str(base)+'.guards.u8','actual_guard_bytes':len(guards),'guard_sha256':hashlib.sha256(guards).hexdigest(),'snapshots':len(rows),'layout_sha256':hashlib.sha256(Path(str(base)+'.layout.jsonl').read_bytes()).hexdigest()}
def host(base,name,expected):
 path=Path(str(base)+name);raw=path.read_bytes();assert raw[:256]==raw[-256:]==bytes([165])*256;assert raw[256:-256]==expected;return{'path':str(path),'physical_bytes':len(raw),'actual_guard_bytes':512,'sha256':hashlib.sha256(raw).hexdigest()}
def input_records(prefix,payloads,case,w,epochs,rows,cols,N,P,Q,Rinput):
 prefix.extend((case,w,epochs,rows,cols,N,128))
 for id in range(5):
  b=payloads[('actual_before_training',id)];prefix.extend((id,len(b)))
  if id==0:
   state=arr('I',b);assert state==array.array('I',[lcg(5551212^i)for i in range(w)]);prefix.extend(state)
  elif id==1:
   wanted=(rates if case<2 else struct.pack('<5f',f32(.0125),0,0,0,0));assert b[:len(wanted)]==wanted;prefix.extend(arr('f',b[:len(wanted)]))
  elif id==2:
   assert b==Rinput
   for node in struct.iter_unpack('<iif',b):prefix.extend(node)
  else:assert b==(P if id==3 else Q);prefix.extend(arr('H',b))
def rng_outputs(out,payloads,w,epochs,updates,vector_size,Rinput,rows,cols):
 trace=arr('q',payloads[('actual_selector_trace',1000)]);states=[lcg(5551212^i)for i in range(w)];nodes=list(struct.iter_unpack('<iif',Rinput));record=[0]*len(nodes);users=[0]*rows;items=[0]*cols;assert len(trace)==w*epochs*updates
 for epoch in range(epochs):
  for step in range(updates):
   for worker in range(w):
    states[worker]=lcg(states[worker]);start=int(f32(f32(f32(float(states[worker]))/f32(float(1<<31)))*f32(len(nodes))))%len(nodes);assert trace[(epoch*updates+step)*w+worker]==start
    for j in range(vector_size):
     row=(start+j)%len(nodes);record[row]+=1;users[nodes[row][0]]+=1;items[nodes[row][1]]+=1
 assert arr('I',payloads[('release',0)])==array.array('I',states);out.extend(trace);out.extend((1,1));out.extend(states);out.extend(record);out.extend(users);out.extend(items);return{'draws':len(trace),'rating_updates':sum(record),'actual_final_states_exact':True,'all_source_draws_exact':True}
def science(path):
 b=path.read_bytes();h=struct.unpack_from('<8i',b);assert h[0]==0x4d464d31 and h[1:4]==(m,n,k)and h[5:]==(30,nnz,nt);off=68;d={}
 for key,N,width in [('halfP',m*k,2),('halfQ',n*k,2),('beforeP',m*k,4),('beforeQ',n*k,4),('finalP',m*k,4),('finalQ',n*k,4),('predTrain',nnz,8),('predTest',nt,8)]:d[key]=b[off:off+N*width];off+=N*width
 assert off==len(b);return d,hashlib.sha256(b).hexdigest()
reports=[]
for trial in r['models'][a.model]['trials']:
 number=trial['trial'];prefix=array.array('d',common);out=array.array('d');cases=[];host_records=[];model_oracles=[]
 for seg,case in [(1,3),(2,4)]:
  base=Path(a.raw_root)/f'{a.model}_{a.process_id}_trial{number}_case{case}';data,receipt=raw_case(base);before=data[('actual_feature_state_input',1)];assert arr('I',before)==array.array('I',[lcg(5551212^i)for i in range(256)]);prefix.extend((case,256,seg*128,128,source_scale,0));prefix.extend(arr('I',before));states=[];bits=[]
  for i in range(256):
   seed=lcg(5551212^i)
   if i<seg*128:seed=lcg(seed);value=f32(f32(f32(float(seed))/f32(float(1<<31)))*source_scale);bits.append(struct.unpack('<H',H(value))[0])
   states.append(seed)
  payload=array.array('H',bits).tobytes();assert data[('release',0)]==payload and data[('release',1)]==array.array('I',states).tobytes();out.extend((case,1,seg*128));out.extend(bits);out.extend(states);out.extend((1,0,0,1));host_records.append(host(base,'.host_features.bin',payload));cases.append(dict(case=case,**receipt))
 base=Path(a.raw_root)/f'{a.model}_{a.process_id}_trial{number}_case2';data,receipt=raw_case(base);input_records(prefix,data,2,12,1,257,257,257,unitP,unitQ,unitR);actual_operator=Path(str(base)+'.operator.bin').read_bytes();assert actual_operator==data[('release',3)]+data[('release',4)];actualP=arr('H',data[('release',3)]);actualQ=arr('H',data[('release',4)]);oldP=arr('H',unitP);oldQ=arr('H',unitQ);max_fraction=0.;max_abs=0.
 for which,row,j,ideal,bound in unit_expect:
  word=(actualP if which==0 else actualQ)[row*128+j];value=struct.unpack('<e',struct.pack('<H',word))[0];error=abs(value-ideal);assert math.isfinite(value)and error<=bound;max_fraction=max(max_fraction,error/bound);max_abs=max(max_abs,error)
 for row in range(257):
  if row not in selected:assert actualP[row*128:(row+1)*128]==oldP[row*128:(row+1)*128]and actualQ[row*128:(row+1)*128]==oldQ[row*128:(row+1)*128]
 assert data[('release',2)]==unitR;out.extend((2,1,3072));rng=rng_outputs(out,data,12,1,1,1,unitR,257,257);out.extend((1,1,0,0));cases.append(dict(case=2,operator_bound_pass=True,independent_components=3072,unchanged_components=62720,max_operator_error=max_abs,max_derived_bound_fraction=max_fraction,rng=rng,**receipt))
 for case,w,updates in [(0,12,53),(1,128,5)]:
  base=Path(a.raw_root)/f'{a.model}_{a.process_id}_trial{number}_case{case}';data,receipt=raw_case(base);input_records(prefix,data,case,w,30,m,n,nnz,initialP,initialQ,R);d,digest=science(Path(str(base)+'.science.bin'));assert data[('release',2)]==R
  for id,key in [(3,'halfP'),(4,'halfQ'),(5,'halfP'),(6,'beforeP'),(7,'halfQ'),(8,'beforeQ')]:assert data[('release',id)]==d[key]
  for suffix,key in [('.host_R.bin',None),('.host_halfP.bin','halfP'),('.host_halfQ.bin','halfQ'),('.host_floatP.bin','finalP'),('.host_floatQ.bin','finalQ')]:host_records.append(host(base,suffix,R if key is None else d[key]))
  argv=[a.model_oracle,str(p/'data/mf_sgd_source_inputs.bin'),str(p/'data/movielens_u1_train.bin'),str(base)+'.science.bin'];completed=subprocess.run(argv,check=True,capture_output=True,text=True);oracle_record=json.loads(completed.stdout);assert oracle_record['pass'];oracle_record['science_sha256']=digest;oracle_record['command']=argv;model_oracles.append(oracle_record);out.extend((case,1,1,1,1,1,1,1));rng=rng_outputs(out,data,w,30,updates,128,R,m,n);out.extend((1,1,2,0));cases.append(dict(case=case,full_scientific_model_and_predictions_pass=True,science_sha256=digest,rng=rng,**receipt))
 assert len(prefix)==1524635;artifact=Path(trial['output_artifact']['saved_path']);actual=artifact.read_bytes();expected=prefix.tobytes()+out.tobytes();assert actual==expected,'Everyactualinput/selector/count/propertyflagmustjoinprimaryexactly';reports.append({'trial':number,'pass':True,'actual_input_prefix_values':len(prefix),'complete_primary_values':len(prefix)+len(out),'primary_sha256':hashlib.sha256(actual).hexdigest(),'all_actual_source_inputs_verified':True,'actual_GPU_guard_bytes':sum(c['actual_guard_bytes']for c in cases),'raw_GPU_bytes':sum(c['raw_bytes']for c in cases),'actual_CPU_host_guard_bytes':sum(h['actual_guard_bytes']for h in host_records),'cases':cases,'host_physical_buffers':host_records,'independent_full_model_oracles':model_oracles})
report={'pass':bool(reports)and all(t['pass']for t in reports),'benchmark':'mf-sgd','model':a.model,'run_id':r['run_id'],'source_result':str(rpath),'source_result_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),'process_id':a.process_id,'checked_trials':len(reports),'numerical_agreement_kind':'property_equivalence_not_identical_raw_values','raw_output_identity_claim':False,'scope':'CompletephysicalactualGPU/CPUguards/payloads joinedtoscientificmodelandprimaryinput/selector/count/propertyrecords. FullHalf->Float32/postscale/unshufflebitexact andeverytrain/testpredictionindependentlyrecomputedLongDouble. All3072wideoperatorcomponentscheckedderivedbound/62720unchangedcomponentsbitexact, bothsourceinitializercapabilitycasesexact. Generalparallelmodelidentity/serialization/convergence notclaimed.','independent_checker_source':str(p/'study_model_oracle.cpp'),'independent_checker_source_sha256':hashlib.sha256((p/'study_model_oracle.cpp').read_bytes()).hexdigest(),'independent_checker_binary_sha256':hashlib.sha256(Path(a.model_oracle).read_bytes()).hexdigest(),'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'trials':reports};Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print({'pass':report['pass'],'trials':len(reports),'GPU_guard_bytes':sum(t['actual_GPU_guard_bytes']for t in reports),'host_guard_bytes':sum(t['actual_CPU_host_guard_bytes']for t in reports),'model_values_checked':sum(sum(x['half_to_float_values_checked']+x['scale_unshuffle_values_checked']for x in t['independent_full_model_oracles'])for t in reports),'prediction_values_checked':sum(sum(x['prediction_values_checked']for x in t['independent_full_model_oracles'])for t in reports)})
