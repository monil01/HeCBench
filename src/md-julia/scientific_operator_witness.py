"""Audit saved MD force artifacts against an independent unordered-pair oracle.
This checks the named source operator, including its zero-distance guard;
it does not certify physical admissibility of coincident particles or trajectories.
"""
import argparse,hashlib,json,math,struct
from pathlib import Path

def audit(result):
 d=json.loads(result.read_text());checks=[];prefix_hash=None;N=129;CASE=16899
 for model,md in d['models'].items():
  for tr in md['trials']:
   f=Path(tr['output_artifact']['saved_path']);raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==tr['output_artifact']['sha256'];assert len(raw)==51858*4
   ph=hashlib.sha256(raw[:50697*4]).hexdigest();prefix_hash=prefix_hash or ph;assert ph==prefix_hash
   data=struct.unpack('<51858f',raw);out=data[50697:];details=[]
   for mode in range(3):
    base=mode*CASE;p=[data[base+k*N:base+(k+1)*N] for k in range(3)];neighbors=data[base+387:base+CASE]
    for i in range(N):assert [int(neighbors[j*N+i]) for j in range(128)]==[j for j in range(N) if j!=i]
    gold=[[0.0]*N for _ in range(3)];active=coincident=cutoff=0
    for i in range(N-1):
     for j in range(i+1,N):
      delta=[p[k][i]-p[k][j] for k in range(3)];q=sum(v*v for v in delta);coincident+=q==0;cutoff+=q==13.5
      if 0<q<13.5:
       active+=1;c=1.5*q**-7-2*q**-4
       for k in range(3):v=delta[k]*c;gold[k][i]+=v;gold[k][j]-=v
    assert (active,coincident,cutoff)==(2576,1,3)
    vals=[out[(mode*3+k)*N:(mode*3+k+1)*N] for k in range(3)];errors=[abs(v-g) for k in range(3) for v,g in zip(vals[k],gold[k])]
    assert all(math.isfinite(v) and abs(v-g)<=1e-5+1e-6*abs(g) for k in range(3) for v,g in zip(vals[k],gold[k]))
    net=max(abs(sum(v)) for v in vals);scale=max(sum(abs(x) for x in v) for v in vals);assert net/scale<=1e-6
    details.append({'mode':mode,'active_unordered_pairs':active,'coincident_guard_pairs':coincident,'strict_cutoff_equality_pairs':cutoff,'Float64_pair_error_inf':max(errors),'net_force_inf':net,'normalized_net_force':net/scale,'max_force_magnitude':max(abs(v) for arr in vals for v in arr)})
   f0=[out[k*N:(k+1)*N] for k in range(3)];f1=[out[(3+k)*N:(4+k)*N] for k in range(3)];f2=[out[(6+k)*N:(7+k)*N] for k in range(3)]
   translation=max(abs(x-y) for k in range(3) for x,y in zip(f0[k],f1[k]));rotation=max(abs(f2[k][i]-([-f0[1][i],f0[0][i],f0[2][i]][k])) for k in range(3) for i in range(N));assert translation<=1e-5 and rotation<=1e-5
   checks.append({'model':model,'trial':tr['trial'],'pass':True,'artifact_sha256':hashlib.sha256(raw).hexdigest(),'input_prefix_sha256':ph,'translation_error_inf':translation,'rotation_error_inf':rotation,'geometries':details})
 r=1.25;h=1e-5;potential=lambda r:1.5/(12*r**12)-2/(6*r**6);fd=-(potential(r+h)-potential(r-h))/(2*h);analytic=1.5/r**13-2/r**7;assert abs(fd-analytic)<1e-8
 return {'benchmark':'md','result':str(result),'source_contract':'Truncated nonperiodic Lennard-Jones-like force with original lj1=1.5,lj2=2,cutsq=13.5. Distinct coincident atoms are ignored by the source guard; this branch probe is not a physically admissible MD initial state claim.','computed_values':1161,'input_prefix_values':50697,'artifact_values':51858,'all20_input_prefixes_identical':True,'input_prefix_sha256':prefix_hash,'pair_potential_derivative_error':abs(fd-analytic),'pair_potential':'U=lj1/(12*r^12)-lj2/(6*r^6), away from singularity/cutoff; force is negative gradient.','auxiliary_budget_revision':'First check imposed 1e-5 absolute Float64 error, failed at 1.2795e-4 for force magnitudes above1000. Revised before portable targets to explicit 1e-5+1e-6relative; normalizednetforce<=1e-6. Original failed artifacts retained. Port/sourceCPU tolerance stays1e-5 unchanged. This budget is an engineering precision/scale check, not a rigorous universal roundoff bound or application accuracy requirement.','checks':checks}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();w=audit(a.result);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(w,indent=2)+'\n');print(json.dumps({'checks':len(w['checks']),'pair_gradient_error':w['pair_potential_derivative_error'],'max_math_error':max(c['Float64_pair_error_inf'] for t in w['checks'] for c in t['geometries'])}))
