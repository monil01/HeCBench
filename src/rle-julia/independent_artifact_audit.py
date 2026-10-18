"""Full byte audit with independently grouped and reconstructed RLE runs."""
from pathlib import Path
import array,struct,itertools,json,hashlib
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;rp=R/'.porting-state/portable_julia/runs/20261018T014244556829Z_rle/result.json';d=json.loads(rp.read_text());raw=(P/'fixture.bin').read_bytes();nc,=struct.unpack_from('<i',raw);off=4;prefix=array.array('i');outputs=array.array('i');desc=[]
for z in range(nc):
 n,kind,entropy,maxseg=struct.unpack_from('<4i',raw,off);off+=16;a=array.array('i');a.frombytes(raw[off:off+4*n]);off+=4*n
 if kind:prefix.extend(a)
 else:assert all(v==1 for v in a)
 keys=[];lengths=[];cursor=0
 for key,group in itertools.groupby(a):
  length=sum(1 for _ in group);assert a[cursor:cursor+length]==array.array('i',[key])*length;cursor+=length;keys.append(key);lengths.append(length)
 assert cursor==n and sum(lengths)==n
 outputs.extend(keys+[0]*(n-len(keys))+[1597463007]*16);outputs.extend(lengths+[0]*(n-len(lengths))+[1597463007]*16);outputs.extend([len(keys)]+[1597463007]*16)
 desc.append({'case':z,'n':n,'num_runs':len(keys),'min_length':min(lengths),'max_length':max(lengths),'logical_input_bytes':4*n,'logical_compact_run_payload_bytes_including_count':4*(2*len(keys)+1),'logical_payload_ratio':(2*len(keys)+1)/n,'allocated_public_key_length_capacity_bytes':8*(n+16),'constant_native_iterator_no_input_array':not kind,'ratio_is_not_physical_traffic_or_actual_file_compression':True})
assert off==len(raw);assert outputs.tobytes()==(P/'expected_i32.bin').read_bytes();want=prefix.tobytes()+outputs.tobytes();assert len(prefix)==5001031 and len(outputs)==12002405
checks=[]
for m,v in d['models'].items():
 assert v['complete']and len(v['trials'])==5
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);assert f.read_bytes()==want;checks.append({'model':m,'trial':t['trial'],'artifact':str(f.relative_to(R)),'sha256':t['output_artifact']['sha256'],'exact_actual_pointer_prefix_values':5001031,'exact_complete_output_tail_guard_values':12002405,'keys_lengths_count_and_tailguards_all_exact':True})
report={'result':str(rp.relative_to(R)),'result_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'all20_exact':True,'pointer_input_prefix_values':len(prefix),'complete_output_tail_guard_values':len(outputs),'checks':checks,'case_descriptors':desc,'oracle':'Independentorderedgroupby/decompressionandbyte-exactcompletearrays. Nativeconstantiteratorlogical1 vsactualJuliaonesarray separatelyassertedbeforewarmup; fullconstantarraynotclaimedasactualnativepointerprefix.','work_count_caveat':'Encodedlogicalpayloadratio describesnamedinputs only. Vendor/custominternaltraffic andnativeconstantiteratorloadworkdiffer; sourcebytesnotmeasuredDRAMbandwidth.'};(P/'independent_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n');print({'all20_exact':True,'prefix':len(prefix),'outputs':len(outputs),'logical_payload_ratios':[x['logical_payload_ratio']for x in desc]})
