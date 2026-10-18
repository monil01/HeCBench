"""Read-only full saved-output/source/PTX audit; never launches GPU work."""
from pathlib import Path
import hashlib,json,re,struct,sys
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
rp=Path(sys.argv[1]) if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261018T044721940181Z_cmembench/result.json'
r=json.loads(rp.read_text());records=[];sources=[];ptx=[]
expected=[]
for fixture in (1,2):
 data=[1]*1024 if fixture==1 else [(j*17)%23-11 for j in range(1024)]
 for w in (1,2,4):expected+=data+[w,1024,4194304,10,fixture]
expected += [1024]*3+[1]*3+[1515870810]*192
assert len(expected)==6372
expected_bytes=struct.pack('<6372i',*expected)
for m,v in r['models'].items():
 assert len(v['trials'])==5 and v['provenance_valid'] and v['timing_valid']
 for trial in v['trials']:
  a=trial['output_artifact'];f=Path(a['saved_path']);data=f.read_bytes();assert data==expected_bytes
  assert hashlib.sha256(data).hexdigest()==a['sha256']
  records.append(dict(model=m,trial=trial['trial'],path=str(f.relative_to(R)),sha256=a['sha256'],all6372_Int32_exact=True,input_prefix_values=6174,checksums=[1024]*3+[1]*3,guards_exact=192))
 for f,h in v['adapter_file_hashes'].items():
  f=Path(f);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;sources.append(dict(model=m,path=str(f),sha256=h))
 if m!='cuda':
  logs=list((R/'.porting-state/portable_julia/runs').glob(f'*_cmembench/{m}_run1.log'));assert len(logs)==1
  log=logs[0];text=log.read_text();parts=re.findall(r'CMEM_PTX_BEGIN width=(\d+)(.*?)CMEM_PTX_END',text,re.S);assert len(parts)==3
  for width,body in parts:
   width=int(width);op={4:'ld.const.b32',8:'ld.const.v2.b32',16:'ld.const.v4.b32'}[width]
   assert op in body and re.search(r'\.const.*study_constant_data\[4096\]',body)
   ptx.append(dict(model=m,width_bytes=width,load=op,constant_bytes=4096,log=str(log.relative_to(R)),sha256=hashlib.sha256(log.read_bytes()).hexdigest(),proof_scope='PTX addressspace/vectorwidth only; not final SASS dynamic work or traffic'))
assert r['matched_performance_valid'] and all(r['variant_verified'].values())
result=dict(result=str(rp.relative_to(R)),all20_complete_artifacts_exact=True,independent_oracle='Python integer sums and complete deterministic input/guard construction, no copied GPU kernel reduction',artifact_records=records,measured_source_files=sources,generated_PTX_evidence=ptx,native_SASS_evidence='native_work_observability.json',limits=['Full observable checksum does not prove all source workers perform reads.','Native actual SASS exits nonwriters before reads for all widths.','Julia final SASS unreviewed; no source-load-volume or bandwidth claim.'])
(P/'saved_output_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(records=len(records),source_files=len(sources),PTX_width_proofs=len(ptx))))
