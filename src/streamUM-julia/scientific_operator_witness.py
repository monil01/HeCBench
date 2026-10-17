#!/usr/bin/env python3
"""Independent exact integer oracle and per-processor-layout counterexample."""
from pathlib import Path
from array import array
import argparse,json,hashlib,sys
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.result.read_text());assert sys.byteorder=='little';prefix=array('d');outputs=array('d');cases=[]
for task,n in enumerate([64,99,100,101,257,511,67,127]):
 # Construct storage independently of GPU thread mapping and Float64 reductions.
 matrix=[(13*(k%n)+7*(k//n)+3*task)%31-15 for k in range(n*n)]
 vector=[(5*j+3*task)%17-8 for j in range(n)]
 gold=[]
 for i in range(n):
  indices=range(i*n,(i+1)*n) if n<100 else range(i,n*n,n)
  gold.append(sum(matrix[k]*vector[j] for j,k in enumerate(indices))/512)
 prefix.extend((n,task,int(n>=100)));prefix.extend(v/32 for v in matrix);prefix.extend(v/16 for v in vector);prefix.extend([0]*n);outputs.extend(gold)
 cases.append({'task':task,'n':n,'processor':'cpu' if n<100 else 'gpu','storage_interpretation':'row_major' if n<100 else 'column_major','gpu_threads':None if n<100 else 256,'gpu_blocks':None if n<100 else (n+255)//256,'exact_integer_numerator_bound':n*15*8,'nonzero_output_count':sum(v!=0 for v in gold)})
assert len(prefix)==384562 and len(outputs)==1326
expected=prefix.tobytes()+outputs.tobytes();digest=hashlib.sha256(expected).hexdigest();rows=[]
for model,md in d['models'].items():
 for t in md.get('trials',[]):
  raw=Path(t['output_artifact']['saved_path']).read_bytes();assert raw==expected;assert hashlib.sha256(raw).hexdigest()==t['output_artifact']['sha256']==digest
  assert len(t['worker_assignments'])==8 and all(0<=w<4 for w in t['worker_assignments'])
  rows.append({'model':model,'trial':t['trial'],'artifact_sha256':digest,'worker_assignments':t['worker_assignments'],'exact_outputs':1326})
# Same bytes A=[1,2,3,4], x=[1,0] mean distinct matrices under the two paths.
row=[1,3];column=[1,2];assert row!=column
r={'benchmark':'streamUM','result':str(a.result.resolve()),'artifact_bytes':len(expected),'artifact_sha256':digest,'actual_output_count':1326,'input_prefix_count':384562,'all_saved_artifacts_exact':True,'artifact_count':len(rows),'rows':rows,'cases':cases,'oracle':'Arbitraryprecisionintegerdotproductnumerators dividedby512. Bound<=61320, henceallproducts/partialsumsexactFloat64 evenwithFMA/reductionreordering. NoGPUindexingorfloataccumulationreused.','layout_witness':{'linear_A':[1,2,3,4],'x':[1,0],'cpu_row_result':row,'gpu_column_result':column,'meaning':'SourceexplicitrowmajorCPUhelper andCUBLAS_OP_N columnmajorGPU. Preservedastask-specificcontract; doesnotcertifyprocessor-independentmeaningofsamebytes.'},'source_binding':'Actual managedowners CPUwrittenbeforetrial, rawdeviceviewsalias sameallocation andheldalive; noimplicitCUDAjlUnifiedMemoryprefetch. Prefixcanonicalbytes supplementedbysource/pointeraudit, notfreshGPUinputreadback.','scope':'Eightnamedtasks,fourlogicaldynamicworkers/fourGPUstreams,3CPU+5GPU. No original1000/4000randomCLI certification, migrationcounters, placementcausality or crossvendorclaim.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'artifact_count':len(rows),'all_exact':True,'sha256':digest,'cases':len(cases)}))
