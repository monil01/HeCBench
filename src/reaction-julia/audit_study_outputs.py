"""Read every saved trial field; separate oracle accuracy and paired agreement."""
from pathlib import Path
import json,struct,hashlib,math,sys
p=Path(__file__).resolve().parent;bundle=Path(sys.argv[1]);d=json.loads((bundle/'result.json').read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();pre=struct.unpack('<90141d',prefix);pos=1;inputs=[]
for c in meta['cases']:
 N=c['N'];pos+=7;inputs.append((pre[pos:pos+N],pre[pos+N:pos+2*N]));pos+=2*N
assert pos==len(pre)
rows=[]
for model,m in d['models'].items():
 for trial in m['trials']:
  f=Path(trial['output_artifact']['saved_path']);raw=f.read_bytes();v=struct.unpack('<'+str(len(raw)//8)+'d',raw);assert raw[:len(prefix)]==prefix and all(map(math.isfinite,v));pos=len(pre);cases=[]
  for c,(ina,inb) in zip(meta['cases'],inputs):
   N=c['N'];gold=struct.unpack('<'+str(9*N)+'d',(p/f'oracle_case{c["case"]}.bin').read_bytes());actual=v[pos:pos+9*N];pos+=9*N;metrics=[]
   for k,name in enumerate(c['oracle_fields']):
    a=actual[k*N:(k+1)*N];r=gold[k*N:(k+1)*N];errs=[abs(x-y) for x,y in zip(a,r)];budget=[1e-6+5e-6*abs(z) for z in r];assert all(e<=b for e,b in zip(errs,budget));maxref=max(map(abs,r));ma=max(errs)
    metrics.append(dict(field=name,max_abs_error=ma,max_abs_reference=maxref,max_error_over_declared_budget=max(e/b for e,b in zip(errs,budget)),error_over_field_max=ma/maxref if maxref else None,max_relative_error_nonzero_reference=max((e/abs(z) for e,z in zip(errs,r) if z!=0),default=None),nonzero_error_at_zero_reference=sum(e!=0 and z==0 for e,z in zip(errs,r)),absolute_floor_weak_for_entire_field=maxref<1e-6))
   aa=actual[:N];bb=actual[N:2*N];assert min(aa)>=0 and min(bb)>=0
   if c['case']==3:assert actual==gold
   gota=v[pos:pos+N];pos+=N;gotb=v[pos:pos+N];pos+=N
   assert struct.pack('<'+str(N)+'f',*gota)==struct.pack('<'+str(N)+'f',*ina);assert struct.pack('<'+str(N)+'f',*gotb)==struct.pack('<'+str(N)+'f',*inb)
   guards=v[pos:pos+144];pos+=144;assert all(z==-12345 for z in guards)
   mda=sum(actual[7*N:8*N])/N;mdb=sum(actual[8*N:9*N])/N;assert abs(mda)<=1e-6 and abs(mdb)<=1e-6
   mass=(sum(aa)-sum(ina))/N
   if c['case']==4:assert all(z==0 for z in bb) and abs(mass)<=1e-6 and min(aa)>=min(ina)-1e-6 and max(aa)<=max(ina)+1e-6
   cases.append(dict(case=c['case'],kind=c['kind'],fields=metrics,mean_diffusion_A=mda,mean_diffusion_B=mdb,mean_A_change=mass,source_timestep_count=c['steps'],exact_equilibrium=c['case']==3,full_actual_initial_input_bits_and_guards_exact=True,final_A_min=min(aa),final_A_max=max(aa),final_B_min=min(bb),final_B_max=max(bb),time_level_note='A/BafterlastEulerupdate;derivatives/rates/diffusionsfrompre-last-updatestate,asoriginalsourcepipeline.'))
  assert pos==len(v)==586333;rows.append(dict(model=model,trial=trial['trial'],artifact_sha256=hashlib.sha256(raw).hexdigest(),values=len(v),cases=cases,pass_=True))
assert len(rows)==20
for x in rows:x['pass']=x.pop('pass_')
a=dict(all_pass=True,rows=rows,artifact_words=586333,sha_classes=len({x['artifact_sha256'] for x in rows}),max_oracle_abs_error=max(f['max_abs_error'] for x in rows for c in x['cases'] for f in c['fields']),max_normalized_budget=max(f['max_error_over_declared_budget'] for x in rows for c in x['cases'] for f in c['fields']),metric_definitions='Programmax_abscomputedoverall9finalfieldsversusindependentF64;max_relnulluncomputed. Hereperfieldrelativeexcludeszeroreferenceexplicitly; fieldnormerrornotglobalerrorbound. Runnerdualreferencemetricsretainedseparately.',scope='Full9finalfields/actualinitialA-Binputbits/guards. CPUF64persteptrajectoryisoracleonly, notcapturedGPUhistory. Engineeringfloatingbudgetnotphysicalmodelerror; nearzerofloors/zero-onlyfieldsvisible inmetrics.')
(p/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:a[k] for k in ['all_pass','sha_classes','max_oracle_abs_error','max_normalized_budget']}))
