#!/usr/bin/env python3
"""Full exact-rational source-quadratic inclusion check; never executes GPU."""
from pathlib import Path
import argparse,json,hashlib,mmap,struct,array
N=65536;D=128
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sign(x,i):
 n,d=x.as_integer_ratio();v=65536*n*n-(131072+i)*n*d+65536*d*d
 return (v>0)-(v<0)
def contains(lo,up,i):
 if i==0:return (lo<=1<=up,lo<=1<=up)
 lower=(lo<=1 and up>=0 and (lo<=0 or sign(lo,i)>=0) and (up>=1 or sign(up,i)<=0))
 upper=(up>=1 and (lo<=1 or sign(lo,i)<=0) and sign(up,i)>=0)
 return lower,upper
ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);ap.add_argument('--raw',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();rp=Path(args.result);r=json.loads(rp.read_text());folder=Path(args.raw);receipt={'result':str(rp),'result_sha256':sha(rp),'audit_source_sha256':sha(Path(__file__)),'primitive_source_witness':'src/interval-julia/rounding_witness.json','criterion':'Strict everyinterval exactrationalcontainment ofoneanalyticquadraticroot andunioncoverageofallroots, no tolerance. ShapePASS doesnot implyconservativeintervalenclosure.','models':{},'counterexamples':[]}
for model,rec in r['models'].items():
 if not rec.get('complete'):continue
 rows=[]
 for trial in range(1,6):
  files=sorted(folder.glob(f'{model}_*_trial{trial}_choice*.counts.i32'));assert len(files)==2,(model,trial,files)
  artifact=Path(rec['trials'][trial-1]['output_artifact']['saved_path']);primary=array.array('d');primary.frombytes(artifact.read_bytes());prefix=[];values=[]
  for file in files:
   case=int(file.name.split('_choice')[1].split('.')[0]);base=Path(str(file)[:-len('.counts.i32')]);count=array.array('i');count.frombytes(file.read_bytes());assert len(count)==N+64
   e=array.array('d');e.frombytes(base.with_name(base.name+'.echo.f64').read_bytes());assert len(e)==2*N+4+64
   echo=list(e[32:32+2*N+4]);assert all(echo[2*i]==echo[2*i+1]==-i/N for i in range(N));assert echo[2*N]==struct.unpack('f',struct.pack('f',.01))[0] and echo[2*N+1]==4
   prefix.extend(echo+[case,N,1,1024,64]);g=base.with_name(base.name+'.guards.u8').read_bytes();assert len(g)==1792 and all(x==165 for x in g)
   path=base.with_name(base.name+'.intervals.bin');f=path.open('rb');mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);assert len(mm)==(N*D+64)*16;physicalguards=mm[:512]+mm[-512:]+file.read_bytes()[:128]+file.read_bytes()[-128:]+base.with_name(base.name+'.echo.f64').read_bytes()[:256]+base.with_name(base.name+'.echo.f64').read_bytes()[-256:];assert physicalguards==g
   missed=spurious=total=0;hist={};first=[]
   for i in range(N):
    k=count[32+i];assert 0<=k<=D;hist[str(k)]=hist.get(str(k),0)+1;intervals=[struct.unpack_from('<dd',mm,(32+j*N+i)*16)for j in range(k)];values.append(float(k));covered=[False,False]
    for lo,up in sorted(intervals):
     values.extend((lo,up));assert lo<=up
     c=contains(lo,up,i);covered=[a or b for a,b in zip(covered,c)];total+=1
     if not any(c):
      spurious+=1
      if len(first)<12:first.append({'equation':i,'interval_hex':[lo.hex(),up.hex()],'contains_analytic_roots':list(c),'polynomial_endpoint_signs':[sign(lo,i),sign(up,i)]})
    if not all(covered):
     missed+=1
     if len(first)<12:first.append({'equation':i,'coverage':covered,'all_actual_intervals_hex':[[l.hex(),u.hex()]for l,u in intervals]})
   row={'trial':trial,'choice':case,'equations':N,'valid_intervals':total,'count_histogram':hist,'missing_root_equations':missed,'spurious_intervals':spurious,'strict_enclosure_pass':missed==0 and spurious==0,'actual_guard_bytes':len(g),'files':[{ 'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}for p in [file,path,base.with_name(base.name+'.echo.f64'),base.with_name(base.name+'.guards.u8')]]};rows.append(row)
   if first:receipt['counterexamples'].append({'model':model,'trial':trial,'choice':case,'examples':first})
   mm.close();f.close()
  assert len(prefix)==r['input_prefix_count'];assert list(primary)==prefix+values,'all retainedactual values mustjoinprimaryartifact exactly'
 receipt['models'][model]={'cases':rows,'strict_enclosure_pass':all(x['strict_enclosure_pass']for x in rows),'equations_checked':sum(x['equations']for x in rows),'complete_output_and_guard_join':True}
Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({m:{k:v for k,v in x.items()if k!='cases'}for m,x in receipt['models'].items()}));print('counterexample_groups',len(receipt['counterexamples']))
