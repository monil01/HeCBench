"""Independent coordinate pullback, transpose scatter, and exact adjoint identity."""
from pathlib import Path
import json,sys,struct,hashlib
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());prefix=[];expected=[];reports=[];case_reports=[]
for B,C,H,W in [(2,3,7,11),(1,2,32,33)]:
 S=B*C*H*W;xn=[(i*13)%67-33 for i in range(S)];gn=[(i*17)%71-35 for i in range(4*S)];prefix.extend(x/16 for x in xn);prefix.extend(x/16 for x in gn);prefix.extend([B,C,H,W,3]);yn=[];dxn=[0]*S
 for q,g in enumerate(gn):
  bc,rem=divmod(q,4*H*W);h,w=divmod(rem,2*W);parent=(bc*H+h//2)*W+w//2;yn.append(xn[parent]);dxn[parent]+=g
 lhs=sum(y*g for y,g in zip(yn,gn));rhs=sum(x*g for x,g in zip(xn,dxn));assert lhs==rhs
 for _ in range(9):expected.extend(x/16 for x in yn)
 for _ in range(9):expected.extend(x/16 for x in dxn)
 case_reports.append({'shape':[B,C,H,W],'all_forward_pixels':4*S,'all_backward_gradients':S,'adjoint_inner_products_exact_integer_numerator':lhs,'inner_product_denominator':256,'legacy_forward_checked_values':B*C*(H//2)*(W//2),'source_launch_settings':18})
complete=tuple(prefix+expected);assert len(complete)==128710
for model,m in r['models'].items():
 for tr in m['trials']:
  p=Path(tr['output_artifact']['saved_path']);raw=p.read_bytes();v=struct.unpack('<128710f',raw);assert v==complete,(model,tr['trial']);reports.append({'model':model,'trial':tr['trial'],'exact_full_output_match':True,'output_values':115830,'prefix_values':12880,'sha256':hashlib.sha256(raw).hexdigest()})
w={'benchmark':'upsample','result':str(rp),'captured_trials_checked':len(reports),'method':'Independent output-coordinate floor mapping for forward, scatter transpose for backward, exact arbitrary-precision integer adjoint identity <Ux,g>=<x,U*g>. Not shared 2x2 GPU checker code.','cases':case_reports,'source_checker_limitations':['Forward source checks B*C*(H/2)*(W/2) instead of4*B*C*H*W; unexamined suffix not certified by legacy PASS.','fabs comparison alone would accept NaN actual outputs; this witness requires exact finite full arrays.'],'scope':'Two named finite dyadic fixtures and all18source launch settings, not arbitrary conditioning/near-overflow precision or large-size throughput.','reports':reports};dest=Path('.porting-state/portable_julia/scientific_witnesses_300/upsample.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w))
