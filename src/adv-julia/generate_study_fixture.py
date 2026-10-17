from pathlib import Path
import json,hashlib,math,struct
from fractions import Fraction as F
p=Path(__file__).resolve().parent
# Exact rational nodal interpolation and differentiation coefficients, rounded once.
s=[F(i-4,4) for i in range(8)];t=[F(i-8,8) for i in range(16)]
def lagrange(nodes,j,x):
 val=F(1)
 for k,z in enumerate(nodes):
  if k!=j:val*=(x-z)/(nodes[j]-z)
 return val
I=[float(lagrange(s,j,x)) for j in range(8) for x in t];weights=[1/math.prod([x-z for k,z in enumerate(t) if k!=j]) for j,x in enumerate(t)]
D=[]
for i,x in enumerate(t):
 row=[F(0) if i==j else weights[j]/(weights[i]*(x-y)) for j,y in enumerate(t)];row[i]=-sum(row);D.extend(float(z) for z in row)
E=2;vgeo=[0.0]*(E*512*12);cg=[0.0]*(E*4096*12);G=[[[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(1)]],[[F(1),F(1,8),F(-1,4)],[F(1,4),F(3,2),F(1,8)],[F(-1,8),F(1,4),F(3,4)]]]
def det(m):return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
ids=[[0,1,7],[2,3,8],[9,10,11]]
for e in range(E):
 dg=det(G[e]);assert dg>0
 for q in range(512):vgeo[e*512*12+6*512+q]=float(dg)
 for q in range(4096):
  for d in range(3):
   for v in range(3):cg[e*4096*12+ids[d][v]*4096+q]=float(G[e][d][v])
  cg[e*4096*12+5*4096+q]=float(1/dg)
A=[[F(1,4),F(-1,8),F(1,16)],[F(1,8),F(3,8),F(-1,4)],[F(-1,4),F(1,16),F(1,2)]];constants=[F(1),F(1,2),F(-1,4)];U=[];gold=[];ideal=[]
for kind in [0,1]:
 aa=[[F(0) for _ in range(3)] for _ in range(3)] if kind==0 else A
 uf=[]
 for f in range(3):
  for e in range(E):
   for z in s:
    for y in s:
     for x in s:uf.append(float(constants[f]+aa[f][0]*x+aa[f][1]*y+aa[f][2]*z))
 U.extend(uf)
 for f in range(3):
  for e in range(E):
   # JW*IJW cancels analytically for consistent affine chart and unit nodal weights.
   coeff=[sum(aa[f][d]*G[e][d][v]*constants[v] for d in range(3) for v in range(3))]+[sum(aa[f][d]*G[e][d][v]*aa[v][q] for d in range(3) for v in range(3)) for q in range(3)]
   for k in range(8):
    wt0=sum(F.from_float(I[k*16+j]) for j in range(16));wt1=sum(F.from_float(I[k*16+j])*t[j] for j in range(16))
    for j in range(8):
     ws0=sum(F.from_float(I[j*16+q]) for q in range(8));ws1=sum(F.from_float(I[j*16+q])*t[q] for q in range(8));fs0=sum(F.from_float(I[j*16+q]) for q in range(16));fs1=sum(F.from_float(I[j*16+q])*t[q] for q in range(16))
     for i in range(8):
      wr0=sum(F.from_float(I[i*16+q]) for q in range(8));wr1=sum(F.from_float(I[i*16+q])*t[q] for q in range(8));fr0=sum(F.from_float(I[i*16+q]) for q in range(16));fr1=sum(F.from_float(I[i*16+q])*t[q] for q in range(16))
      gold.append(float(coeff[0]*wr0*ws0*wt0+coeff[1]*wr1*ws0*wt0+coeff[2]*wr0*ws1*wt0+coeff[3]*wr0*ws0*wt1))
      ideal.append(float(coeff[0]*fr0*fs0*wt0+coeff[1]*fr1*fs0*wt0+coeff[2]*fr0*fs1*wt0+coeff[3]*fr0*fs0*wt1))
prefix=vgeo+cg+D+I+U;data=prefix+gold;(p/'study_fixture.bin').write_bytes(struct.pack('<'+'d'*len(data),*data));fixturemeta={'kind':'exact_rational_Lagrange_matrices_and_manufactured_affine_velocity','input_doubles':len(prefix),'output_doubles':len(gold),'file_sha256':hashlib.sha256((p/'study_fixture.bin').read_bytes()).hexdigest(),'input_prefix_sha256':hashlib.sha256(struct.pack('<'+'d'*len(prefix),*prefix)).hexdigest(),'solution_nodes':[float(x) for x in s],'cubature_nodes':[float(x) for x in t],'source_corner_vs_full_projection_max_difference':max(abs(x-y) for x,y in zip(gold,ideal)),'operator_note':'Sourcebackprojection keeps cubature r/s indices0..7; full16x16standardtensorprojection is NOT implemented/claimed. Analyticoracle applies exact sourcecorner moments tolinearvelocityadvectivepolynomial.','original_source_unchanged':True};(p/'study_fixture_metadata.json').write_text(json.dumps(fixturemeta,indent=2)+'\n')
