#!/usr/bin/env python3
"""CPU order/precision control of actual native-CSF rounded contributions.
Not a replay of the unknown GPU atomic linearization order.
"""
import pathlib,argparse,array,struct,math,json,hashlib,sys
p=argparse.ArgumentParser();p.add_argument('--canonical',type=pathlib.Path,required=True);p.add_argument('--diagnosis',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert sys.byteorder=='little';d=json.loads(a.diagnosis.read_text());case=next(x for x in d['cases']if x['name']=='sourceSize4D');worst=case['modes'][0]['worst_strict_ratio_field'];row=worst['row'];arrays=[]
with a.canonical.open('rb')as f:
 magic,n=struct.unpack('<QQ',f.read(16));assert magic==0x4d4d435346494e50 and n==10
 for j in range(n):
  size,kind=struct.unpack('<QQ',f.read(16));v=array.array('f'if kind==4 else'Q');v.frombytes(f.read(size));arrays.append(v)
vals,_,_,ptr1,_,_,idx2,ptr2,_,_=arrays
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
contributions=[];nonzeros=0
for fiber in range(len(ptr2)-1):
 if idx2[fiber]!=row:continue
 tmp=0.0
 for z in range(ptr2[fiber],ptr2[fiber+1]):
  # Source4Dfbr mode0: exact-binary1.5factor, Float32FMA into tmp;
  # then source left-associated Float32 products by2.5 and3.5.
  tmp=f32(float(vals[z])*1.5+tmp);nonzeros+=1
 contributions.append(f32(f32(tmp*2.5)*3.5))
assert nonzeros==worst['nonzeros_in_row'];exact=worst['exact_Float32_input_sum_oracle'];orders={'CSF_fiber_order':contributions,'reverse_CSF_fiber_order':list(reversed(contributions)),'ascending_magnitude':sorted(contributions,key=abs)};rows=[]
for name,items in orders.items():
 accum=0.0
 for v in items:accum=f32(accum+v)
 rows.append({'order':name,'accumulator':'Float32RN','result':accum,'abs_error_vs_exact_COO':abs(accum-exact),'strict_budget':worst['strict_budget'],'strict_gate_pass':abs(accum-exact)<=worst['strict_budget']})
accum=math.fsum(contributions);rows.append({'order':'same_contribution_multiset','accumulator':'Pythonmath.fsumFloat64','result':accum,'abs_error_vs_exact_COO':abs(accum-exact),'strict_budget':worst['strict_budget'],'strict_gate_pass':abs(accum-exact)<=worst['strict_budget']})
out={'benchmark':'mmcsf','canonical':str(a.canonical),'canonical_sha256':hashlib.sha256(a.canonical.read_bytes()).hexdigest(),'target_mode':0,'row':row,'input_nonzeros':nonzeros,'rounded_source_fiber_contributions':len(contributions),'exact_COO_oracle':exact,'retained_native_GPU_output':worst['got'],'retained_native_GPU_error':worst['abs_error'],'controls':rows,'scope':'Actual native canonicalCSF Float32contributiondata; CPUchangesonlyaccumulationorder/precision, notGPUexecution. GPUatomicorderunknown. Float64sumisolatesaccumulationlossfrompre-roundedproduct/fibererror. This is a diagnostic control, not a replacementnative/referencecontract, acceptedfive-trialseries, causalproof oftheexactGPUerror or performanceevidence.'};q=a.output.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(a.output);print(json.dumps(rows))
