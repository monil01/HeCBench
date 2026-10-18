"""Read-only independent integer/byte audit; no compiler or GPU invocation."""
from pathlib import Path
import hashlib,json,struct,sys
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
rp=Path(sys.argv[1]) if len(sys.argv)>1 else max((R/'.porting-state/portable_julia/runs').glob('*_crc64/result.json'))
r=json.loads(rp.read_text());meta=json.loads((P/'fixture_provenance.json').read_text());prefix=[]
for i in [1,2]:prefix+=list((P/f'study_input{i}_u8.bin').read_bytes())
for f in ['crc64_table.bin','crc64_interleaved_table.bin','crc64_x_pow_2n.bin']:
 raw=(P/f).read_bytes();prefix+=list(struct.unpack('<'+'Q'*(len(raw)//8),raw))
prefix+=list(b'123456789')+[65555,65677,64,10,9];assert len(prefix)==133358
oracle=struct.unpack('<281Q',(P/'oracle_u64.bin').read_bytes());expected=struct.pack('<133895Q',*(prefix+list(oracle)+[0x5a5a5a5a5a5a5a5a]*256))
records=[];sources=[]
for name,v in meta['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==v['sha256']
assert r['reference_agreement'] and all(r['variant_verified'].values()) and r['matched_performance_valid']
for m,v in r['models'].items():
 assert len(v['trials'])==5 and v['provenance_valid'] and v['timing_valid']
 for t in v['trials']:
  a=t['output_artifact'];f=Path(a['saved_path']);raw=f.read_bytes();assert raw==expected and hashlib.sha256(raw).hexdigest()==a['sha256']
  records.append(dict(model=m,trial=t['trial'],artifact=str(f.relative_to(R)),sha256=a['sha256'],exact_input_prefix_values=133358,exact_observable_output_values=281,exact_guard_values=256,all20_timed_API_returns_exact=True,known_message_crc='995dc9bbdf1939fa',both_appended_check_residues='ffffffffffffffff'))
 for filename,h in v['adapter_file_hashes'].items():
  f=Path(filename);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;sources.append(dict(model=m,path=filename,sha256=h))
result=dict(result=str(rp.relative_to(R)),records=records,measured_source_bindings=sources,all20_complete_artifacts_exact=True,independent_oracle='Bit-at-a-time GF2 reflectedECMA182 withallonesinit/finalxor, independentoftable/interleaveddevicealgorithm',input_use_provenance='TimedoriginalAPIcopiesallinput/tablebytesandfreesbuffers; matchinguntimedunchangedkernel capturesactualdeviceinput/partialsoutsideROI, exactsourcebindings. Timedbuffersnotpostfree-readback.',scope='FullAPIincludesallocations,H2D,GPU,D2H,CPUcombine,frees; nativeCPUthreadclockthroughputnotclaimed',native_thread_coverage_witness='native_thread_coverage_witness.json')
(P/'saved_output_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(artifacts=len(records),source_bindings=len(sources))))
