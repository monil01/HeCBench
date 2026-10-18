from pathlib import Path
import array,json,math
p=Path('src/hexciton-julia');f=array.array('f');f.frombytes((p/'fixtures_f32.bin').read_bytes());o=array.array('d');o.frombytes((p/'oracle_f64.bin').read_bytes());run=Path('.porting-state/portable_julia/runs/20261018T062142620845Z_hexciton');a=array.array('f');a.frombytes((run/'cuda_trial1_trial_1.bin').read_bytes());i=0;pos=len(f);oi=0;bad=[];guards=True;ro=True
for case in range(75):
 N,mode,kid,dim,reps=map(int,f[i:i+5]);i+=6;h=list(f[i:i+98]);i+=98;sig=list(f[i:i+98*N]);i+=98*N;hb=a[pos:pos+106];pos+=106;sb=a[pos:pos+98*N+8];pos+=98*N+8;ro &=list(hb[4:-4])==h and list(sb[4:-4])==sig
 for z in[hb,sb]:guards &=all(v==-8192.25 for v in z[:4])and all(v==-8192.25 for v in z[-4:])
 if kid==0:out=a[pos:pos+8];pos+=8;guards &=all(v==-8192.25 for v in out);continue
 out=a[pos:pos+98*N+8];pos+=98*N+8;got=out[4:-4];expected=o[oi:oi+98*N];oi+=98*N;guards &=all(v==-8192.25 for v in out[:4])and all(v==-8192.25 for v in out[-4:]);e=max(abs(v-w)for v,w in zip(got,expected));primary=all(abs(v-w)<=1e-6+1e-4*abs(w)for v,w in zip(got,expected))
 if notprimary or(mode==2 and any(v!=0 for v in got)):bad.append({'N':N,'mode':mode,'kid':kid,'primary_tolerance_pass':primary,'max_error':e,'max_abs_output':max(map(abs,got)),'nonzero_count':sum(v!=0 for v in got),'first_nonzero':next(((j,v)for j,v in enumerate(got)if v!=0),None)})
print(json.dumps({'complete_parse':pos==len(a),'allguards':guards,'allreadonly':ro,'criterion_failures':bad},indent=2));(p/'rejected_native_identity_audit.json').write_text(json.dumps({'complete_parse':pos==len(a),'allguards':guards,'allreadonly':ro,'criterion_failures':bad,'raw_result':str(run/'result.json'),'status':'Originalexactzeroauxiliarycriterionrejected; noacceptancerevisionyet'},indent=2)+'\n')
