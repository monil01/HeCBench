"""Exact rational solve of actual rounded inputs plus complex residual checks; no GPU run."""
from pathlib import Path
from fractions import Fraction
import array,json,struct,sys
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());f=json.loads((Path(__file__).parent/'study_fixture.json').read_text())
def rounded(x,single):return struct.unpack('<f',struct.pack('<f',x))[0]if single else x
def solve(A,b):
 m=[list(map(Fraction,row))+[Fraction(v)]for row,v in zip(A,b)];n=len(b)
 for k in range(n):
  pivot=max(range(k,n),key=lambda i:abs(m[i][k]));assert m[pivot][k]!=0;m[k],m[pivot]=m[pivot],m[k];v=m[k][k];m[k]=[x/v for x in m[k]]
  for i in range(n):
   if i==k:continue
   v=m[i][k];m[i]=[x-v*y for x,y in zip(m[i],m[k])]
 return [row[-1]for row in m]
prefix=[];cases=[]
for code,name in enumerate(f['types']):
 single=code<2;complex_type=code in(1,3);a=[rounded(x,single)for x in f['A_column_major_two_batches']];b=[rounded(x,single)for x in f['B_two_batches']]
 for value in a+b:prefix.extend([0.,value]if complex_type else[value,0.])
 prefix.extend([code,5,5,1,2,[4,8,8,16][code]])
 systems=[]
 for batch in range(2):
  A=[[a[batch*25+j*5+i]for j in range(5)]for i in range(5)];B=b[batch*5:(batch+1)*5];X=solve(A,B);invc=[solve(A,[int(i==j)for i in range(5)])for j in range(5)];anorm=max(sum(abs(v)for v in row)for row in A);invnorm=max(sum(abs(float(invc[j][i]))for j in range(5))for i in range(5));systems.append({'A':A,'b':B,'x':[float(v)for v in X],'condition_inf':anorm*invnorm,'exact_solution_to_one_max_abs':max(abs(float(v)-1)for v in X)})
 cases.append({'name':name,'single':single,'complex':complex_type,'systems':systems})
report={'oracle':'Exact Fraction Gaussian elimination with pivoting on actual Float32/Float64 rounded coefficient values; independent from GPU QR and no normal-equation solve.','input_prefix_count':len(prefix),'solution_components':80,'status_info_components':8,'trials':[],'cases':[{'type':c['name'],'condition_inf':[x['condition_inf']for x in c['systems']],'exact_actual_input_solution_to_one_max_abs':[x['exact_solution_to_one_max_abs']for x in c['systems']]}for c in cases],'scope':'Only original square5x5 two-batch systems. These fixtures are not an overdetermined least-squares validation, ill-conditioned stress test or framework-authored solver.'}
for model,spec in r['models'].items():
 for tr in spec.get('trials',[]):
  a=array.array('d');a.frombytes(Path(tr['output_artifact']['saved_path']).read_bytes())
  if sys.byteorder!='little':a.byteswap()
  assert list(a[:504])==prefix,(model,'prefix')
  out=a[504:];assert len(out)==88;checks=[]
  for ci,c in enumerate(cases):
   row=out[ci*22:(ci+1)*22];assert row[0]==0 and row[1]==0
   got=[complex(row[2+2*k],row[3+2*k])for k in range(10)];bound=1e-6 if c['single']else 1e-8
   for batch,system in enumerate(c['systems']):
    x=got[batch*5:(batch+1)*5];error=max(abs(v-w)for v,w in zip(x,system['x']));assert error<bound,(model,c['name'],error)
    A=system['A'];b=system['b'];res=max(abs(sum(complex(A[i][j])*x[j]for j in range(5))-b[i])for i in range(5));den=max(sum(abs(v)for v in row)for row in A)*max(abs(v)for v in x)+max(abs(v)for v in b);backward=res/den
    checks.append({'type':c['name'],'batch':batch,'max_complex_abs_error_vs_exact_actual_input_solve':error,'max_complex_abs_error_to_one':max(abs(v-1)for v in x),'residual_inf':res,'normwise_backward_error_inf':backward,'bound':bound})
  report['trials'].append({'model':model,'trial':tr['trial'],'exact_prefix':True,'checks':checks})
q=rp.parent/'scientific_linear_witness.json';q.write_text(json.dumps(report,indent=2)+'\n');print(q)
