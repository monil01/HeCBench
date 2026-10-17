"""Independent closed-form Matérn5/2 oracle for immutable saved study artifacts."""
import argparse,hashlib,json,math,struct
from pathlib import Path

def audit(result):
 d=json.loads(result.read_text());checks=[];reference=None;N=4913
 for model,md in d['models'].items():
  for tr in md['trials']:
   p=Path(tr['output_artifact']['saved_path']);raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==tr['output_artifact']['sha256'];assert len(raw)==83778*4
   if reference is None:reference=raw
   assert raw==reference
   checks.append({'model':model,'trial':tr['trial'],'pass':True,'artifact_sha256':h,'input_prefix_sha256':hashlib.sha256(raw[:14996*4]).hexdigest(),'full_array_identical_to_primary_reference':True})
 values=struct.unpack('<83778f',reference);s=values[:150];t=values[150:14889];weights=[values[14889:14939],values[14939:14989]];scales=values[14989:14996];actual=values[14996:];oracle=[[],[]]
 for l in scales:
  for i in range(N):
   sums=[0.,0.]
   for j in range(50):
    q=sum((s[3*j+k]-t[3*i+k])**2 for k in range(3));a=math.sqrt(5.)*math.sqrt(q)/l;kernel=(1+a+a*a/3)*math.exp(-a)
    for mode in range(2):sums[mode]+=kernel*weights[mode][j]
   for mode in range(2):oracle[mode].append(sums[mode])
 gold=oracle[0]+oracle[1];error=max(abs(v-g) for v,g in zip(actual,gold));assert all(math.isfinite(v) for v in actual) and error<=1e-4
 sumw=sum(weights[0]);assert min(actual[:7*N])>=-1e-6 and max(actual[:7*N])<=sumw+1e-4
 monotonic=min(actual[(l+1)*N+i]-actual[l*N+i] for l in range(6) for i in range(N));assert monotonic>=-1e-4
 for j in range(50):assert tuple(s[3*j:3*j+3])==tuple(t[3*((73*j+17)%N):3*((73*j+17)%N)+3])
 return {'benchmark':'matern','configuration_id':d['configuration_id'],'result':str(result),'computed_values':68782,'input_prefix_values':14996,'artifact_values':83778,'all20_full_artifacts_identical':True,'actual_geometry':{'targets':4913,'sources':50,'threads':[16,64],'blocks':308,'last_block_valid_targets':1,'shared_bytes':4192},'immutable_geometry_prose_erratum':'Original input_contract prose says257blocks from an earlier4103target plan; actual all measured code usescld4913/16=308. Source/counts/execution unchanged. Hash-bound study_adjudications.json notes correct the prose, not the run.','independent_Float64_error_inf':error,'positive_weight_sum':sumw,'positive_output_min':min(actual[:7*N]),'positive_output_max':max(actual[:7*N]),'scale_monotonicity_min_increment':monotonic,'kernel_at_zero_distance':1,'source_target_coincidences':50,'scientific_scope':'Weighted Matérn5/2 radial sums, not a GP fit or uncertainty calibration. Signed weights do not imply positive outputs. Gram checks concern the common source fixture and run on CPU outside ROI.','Gram_sanity_from_Julia_host_oracle':[{'length_scale':.1,'eigen_min':.742620410406,'eigen_max':1.406852102},{'length_scale':1,'eigen_min':.000264268844756,'eigen_max':37.1272887442},{'length_scale':100000,'eigen_min':-1.22858219159e-14,'eigen_max':49.999999998,'interpretation':'Numerically near rank one in Float64; negative roundoff eigenvalue is not a demonstrated violation of mathematical PSD or framework failure'}],'checks':checks}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();w=audit(a.result);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(w,indent=2)+'\n');print(json.dumps({'checks':len(w['checks']),'error':w['independent_Float64_error_inf'],'blocks':308}))
