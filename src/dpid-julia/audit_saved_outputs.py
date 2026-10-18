"""Read-only complete RGB/input/guard/source audit; no GPU or new timings."""
from pathlib import Path
import hashlib,json,struct,sys
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
configs=json.loads((P/'configuration_results.json').read_text());meta=json.loads((P/'fixture_provenance.json').read_text());records=[];sources=[]
for name,info in meta['files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==info['sha256']
for cfg in configs:
 rp=R/cfg['result'];r=json.loads(rp.read_text());label=cfg['lambda_label'];lam=cfg['lambda_value'];prefix=[];oracle=[]
 for ci,(iw,ih,pw,ph) in enumerate([(40,30,2.5,2.5),(128,96,8,8)],1):
  prefix+=list((P/f'study_input{ci}_u8.bin').read_bytes())+[16,12,iw,ih]+list(struct.unpack('<3I',struct.pack('<3f',pw,ph,lam)))+[10]
  oracle+=list((P/f'oracle_guidance{ci}_u8.bin').read_bytes())+list((P/f'oracle_output{ci}_lambda_{label}_u8.bin').read_bytes())
 assert len(prefix)==40480 and len(oracle)==2304;prefixbytes=struct.pack('<40480I',*prefix);guardbytes=struct.pack('<576I',*([165,90,195]*192))
 assert r['reference_agreement'] and all(r['variant_verified'].values()) and r['matched_performance_valid']
 for model,v in r['models'].items():
  assert len(v['trials'])==5 and v['provenance_valid']
  for row in v['trials']:
   a=row['output_artifact'];f=Path(a['saved_path']);raw=f.read_bytes();assert len(raw)==43360*4 and hashlib.sha256(raw).hexdigest()==a['sha256'];assert raw[:40480*4]==prefixbytes and raw[(40480+2304)*4:]==guardbytes
   values=struct.unpack('<43360I',raw);actual=values[40480:42784];assert all(x<=255 for x in actual);errors=[abs(x-y) for x,y in zip(actual,oracle)];assert max(errors)<=1
   per_case=[]
   for ci in [0,1]:
    start=ci*1152;g=errors[start:start+576];o=errors[start+576:start+1152];per_case.append(dict(case=ci+1,guidance_max_byte_error=max(g),output_max_byte_error=max(o),guidance_mismatched_channels=sum(x!=0 for x in g),output_mismatched_channels=sum(x!=0 for x in o)))
   dual={}
   for ref in ['cuda','cuda_jl']:
    fref=Path(r['models'][ref]['trials'][row['trial']-1]['output_artifact']['saved_path']);rv=struct.unpack('<43360I',fref.read_bytes())[40480:42784];e=max(abs(x-y) for x,y in zip(actual,rv));assert e<=1;dual[ref]=e
   records.append(dict(lambda_value=lam,model=model,trial=row['trial'],artifact=str(f.relative_to(R)),sha256=a['sha256'],input_prefix_exact_values=40480,guards_exact_byte_values=576,complete_RGB_channel_values=2304,independent_max_byte_error=max(errors),cases=per_case,dual_reference_max_byte_errors=dual))
  for filename,h in v['adapter_file_hashes'].items():
   f=Path(filename);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;sources.append(dict(lambda_value=lam,model=model,path=filename,sha256=h))
reference=P/'reference_original/main.jl';current=P/'dpid-julia-cuda.jl/main.jl';assert reference.read_bytes()==current.read_bytes()
result=dict(benchmark='dpid',records=records,measured_source_bindings=sources,all60_complete_artifacts_pass=True,byte_tolerance=1,rtol=0,guards_and_inputs_exact_bytes=True,reference_device_source_byte_preserved=True,original_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),source_mapping_confound='Nativewarp32cooperativeperoutput vs all3Juliaserialoneworkerperoutput',scope_limits=['Bothpatchregimesisotropic2.5square/8square; originalMakefileanisotropic12.8x17.0667unmeasured.','Sourceinspiredone-bytequantizationcontract,notallrawRGBbitsidentical; no claimsoleerrorcausewithoutunquantizedFloat32trace.','No original8192squareimage/cache/image-qualityvalidation.'])
(P/'saved_output_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(dict(artifacts=len(records),source_bindings=len(sources),max_byte_error=max(x['independent_max_byte_error'] for x in records)))
