"""Read-only integer-artifact audit, with CPU/GPU evidence scopes kept distinct."""
from pathlib import Path
import struct,json,hashlib
p=Path(__file__).resolve().parent;root=p.parents[1];bundle=root/'.porting-state/portable_julia/runs/20261018T010921456932Z_secp256k1';r=json.loads((bundle/'result.json').read_text());f=(p/'fixture.bin').read_bytes();gold=(p/'expected_bytes.bin').read_bytes();desc=json.loads((p/'fixture_description.json').read_text());checks=[];timings={};hashes=[]
for model,m in r['models'].items():
 rows=[]
 for t in m['trials']:
  path=Path(t['output_artifact']['saved_path']);data=path.read_bytes();assert data[:len(f)]==f;out=struct.unpack('<147I',data[len(f):]);assert all(x<=255 for x in out);assert bytes(out)==gold
  for c in range(3):assert out[c*49+32:(c+1)*49]==(165,)*17
  h=hashlib.sha256(data).hexdigest();hashes.append(h);checks.append({'model':model,'trial':t['trial'],'sha256':h,'actual_UInt32_input_prefix_exact':True,'all147_output_bytes_exact':True,'all51_guard_bytes_exact':True,'complete32byte_results':[bytes(out[c*49:c*49+32]).hex() for c in range(3)]});rows.append(t['per_path_times_seconds'])
 timings[model]={'all5_path_times_seconds':rows,'mean_path_times_last4_seconds':[sum(row[c] for row in rows[1:])/4 for c in range(3)],'mean_sum_seconds':m['mean_seconds']}
assert hashlib.sha256(f).hexdigest()==desc['fixture_sha256'] and hashlib.sha256(gold).hexdigest()==desc['gold_sha256']
a={'benchmark':'secp256k1','bundle':str(bundle.relative_to(root)),'all20_full_checks':len(checks)==20,'all20_bit_identical':len(set(hashes))==1,'artifact_sha256':hashes[0],'prefix_count':len(f)//4,'output_count':147,'actual_GPU_storage':['UInt32pointlimbs','UInt8outputbytes'],'actual_GPU_arithmetic':['UInt32radix26limbs','UInt64products/carries'],'serialization':'OutputUInt8valueslosslesslywidenedUInt32afterD2H, nofloatingpointGPUclaim.','all_complete_public_outputs':True,'GPU_intermediate_states_captured':False,'CPU_oracle_scope':'Independentarbitrary-integer511stepJacobiancurvechecks/nonzeroH/nonzeroZandinverseidentity; sameaffineendpointacross3orders. Final32outputbytesstrictlycheckedallGPUmodels. CPU44valuefieldunitchecks1936mul/44sqr/43inv areCPU-only additionalmathcodeaudit, notGPUunitseries.','relative_metric':'Allacceptedoutputbytesexact implyabsoluteerror0andanydenominator-floorrelativeerror0; nonzero integererrorwouldbepassfailure.','full_checks':checks,'per_model_timings':timings,'geometry_witness_sha256':hashlib.sha256((p/'independent_geometry_witness.json').read_bytes()).hexdigest()}
fout=p/'independent_artifact_audit.json';tmp=fout.with_suffix('.json.tmp');tmp.write_text(json.dumps(a,indent=2)+'\n');tmp.replace(fout);print({k:a[k] for k in ('all20_bit_identical','artifact_sha256')})
