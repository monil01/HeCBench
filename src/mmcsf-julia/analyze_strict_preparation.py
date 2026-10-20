#!/usr/bin/env python3
"""CPU replay of retained rejected state. Exact integer sums of Float32 inputs."""
import argparse,pathlib,array,sys,math,struct,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--artifact',type=pathlib.Path,required=True);p.add_argument('--fixtures',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert sys.byteorder=='little';v=array.array('d');v.frombytes(a.artifact.read_bytes());assert len(v)==3690812
cases=[];i=0
for case_id,name in enumerate(['toy','control3D','longFiber4D','sourceSize4D']):
 meta=list(v[i:i+7]);i+=7;assert meta[0]==case_id;nd,nz,rank=map(int,meta[1:4]);dims=list(map(int,v[i:i+nd]));i+=nd;sz=sum(dims)*rank;offset=[0]
 for d in dims:offset.append(offset[-1]+d*rank)
 inputs=[]
 for mode in range(nd):
  assert list(v[i:i+2])==[case_id,mode];i+=2;inputs.append(v[i:i+sz]);i+=sz
  for m in range(nd):
   expected=0.0 if m==mode else m+.5
   assert all(x==expected for x in inputs[-1][offset[m]:offset[m+1]])
 cases.append(dict(name=name,nd=nd,nz=nz,rank=rank,dims=dims,offset=offset,sz=sz,inputs=inputs))
assert i==1640339;rows=[];overall_gold_error=0.0
for case_id,c in enumerate(cases):
 D=c['dims'];nd=c['nd'];rank=c['rank'];sums=[[0]*d for d in D];abss=[[0]*d for d in D];counts=[[0]*d for d in D];n=0;f=a.fixtures/(c['name']+'.tns');fixture_sha=hashlib.sha256()
 with f.open('rb')as stream:
  header=next(stream);fixture_sha.update(header);assert int(header)==nd
  header=next(stream);fixture_sha.update(header);assert list(map(int,header.split()))==D
  for line in stream:
   fixture_sha.update(line);fields=line.split();ix=[int(x)-1 for x in fields[:nd]];bits=struct.unpack('<I',struct.pack('<f',float(fields[nd])))[0];e=(bits>>23)&255;assert e!=255;mant=(bits&0x7fffff)+(0x800000 if e else 0);units=mant<<(e-1 if e else 0)
   if bits>>31:units=-units
   for m,row in enumerate(ix):sums[m][row]+=units;abss[m][row]+=abs(units);counts[m][row]+=1
   n+=1
 assert n==c['nz'];mode_rows=[]
 for mode in range(nd):
  assert list(v[i:i+2])==[case_id,mode];i+=2;got=v[i:i+c['sz']];i+=c['sz'];stored_gold=v[i:i+D[mode]*rank];i+=D[mode]*rank
  for m in range(nd):
   if m!=mode:assert got[c['offset'][m]:c['offset'][m+1]]==c['inputs'][mode][c['offset'][m]:c['offset'][m+1]]
  numerator=math.prod(2*m+1 for m in range(nd)if m!=mode);power=-149-(nd-1);maxerr=0.0;maxratio=0.0;maxgold=0.0;fail=source_fail=bound_fail=0;worst=None;maxspread=0.0
  for row in range(D[mode]):
   exact=math.ldexp(float(sums[mode][row]*numerator),power);abs_sum=math.ldexp(float(abss[mode][row]*numerator),power);k=4*counts[mode][row]+16;eps=2**-24;gamma=k*eps/(1-k*eps)if k*eps<1 else math.inf;bound=gamma*abs_sum;outrow=got[c['offset'][mode]+row*rank:c['offset'][mode]+(row+1)*rank];maxspread=max(maxspread,max(outrow)-min(outrow))
   for col,x in enumerate(outrow):
    assert math.isfinite(x);delta=abs(x-exact);budget=.01+2e-5*abs(exact);ratio=delta/budget;gerr=abs(stored_gold[row*rank+col]-exact);maxgold=max(maxgold,gerr);fail+=delta>budget;source_fail+=delta>.1;bound_fail+=delta>bound+1e-10
    if ratio>maxratio:maxratio=ratio;worst=dict(row=row,column=col,got=x,exact_Float32_input_sum_oracle=exact,abs_error=delta,strict_budget=budget,nonzeros_in_row=counts[mode][row],sum_absolute_terms=abs_sum,conservative_gamma_bound=bound)
    maxerr=max(maxerr,delta)
  overall_gold_error=max(overall_gold_error,maxgold);mode_rows.append({'mode':mode,'output_fields':D[mode]*rank,'max_abs_error':maxerr,'max_error_over_strict_budget':maxratio,'strict_budget_failed_fields':fail,'source_0_1_counterfactual_failed_fields':source_fail,'max_stored_longdouble_gold_vs_exact_integer_sum':maxgold,'max_column_output_spread_under_identical_factor_columns':maxspread,'conservative_gamma_bound_failed_fields':bound_fail,'worst_strict_ratio_field':worst})
 rows.append({'name':c['name'],'dimensions':D,'rank':rank,'nonzeros':n,'fixture_sha256':fixture_sha.hexdigest(),'actual_all_premode_factor_fields_match_source_constant_or_zero':True,'modes':mode_rows})
assert i==len(v);out={'benchmark':'mmcsf','artifact':{'path':str(a.artifact),'bytes':a.artifact.stat().st_size,'sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest()},'cases':rows,'method':'Each textvalue parsedPythonFloat64 thenroundedFloat32; exact signed integer units2^-149 summedpermode/row; constant factor products are exact oddinteger/poweroftwo. All actualpreinputcolumnschecked. Every recordedGPUoutput field checked; source0.1 is counterfactual descriptive, not revisedacceptedcontract.','max_original_CPU_longdouble_gold_vs_integer_sum':overall_gold_error,'strict_acceptance_unchanged':True,'source_0_1_counterfactual_pass':all(m['source_0_1_counterfactual_failed_fields']==0 for c in rows for m in c['modes']),'scope':'Rejected preparation only, not5trials or acceptedfourmodelagreement. Conservative gamma_(4*n+16)Float32forwardbound is a loose consistencycheck, not a measured operation trace, physicalerrorbound or definitivecause. No tolerancechange/GPUrerun.'};q=a.output.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(a.output);print(json.dumps({'cases':len(rows),'max_gold_difference':overall_gold_error,'source_0_1_counterfactual_pass':out['source_0_1_counterfactual_pass'],'strict_failed_fields':sum(m['strict_budget_failed_fields']for c in rows for m in c['modes'])}))
