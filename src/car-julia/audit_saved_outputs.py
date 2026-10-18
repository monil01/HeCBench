from pathlib import Path
import json,hashlib,sys,struct
P=Path(__file__).resolve().parent;R=P.parent.parent;rp=R/sys.argv[1];r=json.load(open(rp))
input=list(struct.unpack('<30438f',(P/'study_inputs_f32.bin').read_bytes()))+[2.,3.,19.,23.,9.,37.,29.,1.,1.,100.];oracle=list(struct.unpack('<2622d',(P/'oracle_f64.bin').read_bytes()));rows=[];payload={}
for m,mr in r['models'].items():
 assert len(mr['trials'])==5
 for tr in mr['trials']:
  raw=Path(tr['output_artifact']['saved_path']).read_bytes();v=list(struct.unpack('<33230f',raw));assert v[:30448]==input;assert raw[:30448*4]==(P/'study_inputs_f32.bin').read_bytes()+struct.pack('<10f',2,3,19,23,9,37,29,1,1,100);assert raw[-160*4:]==struct.pack('<160f',*([12345.125]*160));assert v[-160:]==[12345.125]*160;got=v[30448:33070];scaled=max(abs(a-b)/(5e-5+2e-6*abs(b))for a,b in zip(got,oracle));assert scaled<=1;payload[m,tr['trial']]=got;rows.append(dict(model=m,trial=tr['trial'],artifact_sha256=hashlib.sha256(raw).hexdigest(),exact_30448_value_input_prefix=True,exact_160_guards=True,max_scaled_independent_closedform_error=scaled))
comparisons=[]
for target in ['cuda_jl','ka','jacc']:
 for reference in ['cuda','cuda_jl']:
  if target==reference:continue
  for trial in range(1,6):
   a=payload[target,trial];b=payload[reference,trial];scaled=max(abs(x-y)/(5e-5+2e-6*abs(y))for x,y in zip(a,b));assert scaled<=1;comparisons.append(dict(target=target,reference=reference,trial=trial,max_scaled_output_error=scaled))
report=dict(result=str(rp.relative_to(R)),result_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),all20_complete_outputs_inputbits_and_guards_pass=True,all5_dualrefs_strict_payload_pass=True,oracle='Independentclosedformbilinearpolynomial withsourceeffectiveclampedneighborcoordinates',tolerance={'atol':5e-5,'rtol':2e-6},rows=rows,comparisons=comparisons,historical_rejected_result='.porting-state/portable_julia/runs/20261018T041719231496Z_car/result.json',historical_checker_counterexample=json.load(open(P/'reference_selfcheck_witness.json')));(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['rows','comparisons']}))
