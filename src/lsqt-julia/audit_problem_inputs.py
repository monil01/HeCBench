#!/usr/bin/env python3
"""Independent source-problem structural/Hermitian/normalization audit, no GPU."""
from pathlib import Path
import array,cmath,hashlib,json,math,struct
p=Path(__file__).resolve().parent;records=[]
for name in ['diffusive','localized','small_lattice','general_tail','spin_pairs']:
 file=p/(name+'_fixture.bin');f=file.open('rb');h=struct.unpack('<12i',f.read(48));n,maxn,ne,nt,nm,rv,v0,vac,msd,spin,ldos,nl=h;volume,emax=struct.unpack('<2d',f.read(16))
 def read(T,N):
  a=array.array(T);a.frombytes(f.read(N*a.itemsize));assert len(a)==N;return a
 energy=read('d',ne);time=read('d',nt);local=read('i',nl);counts=read('i',n);neighbors=read('i',n*maxn);potential=read('d',n);real=read('d',n*maxn);imag=read('d',n*maxn);dx=read('d',n*maxn);states=[(read('d',n),read('d',n))for _ in range(rv)];assert not f.read(1)
 H={};J={};normbound=0.
 for i in range(n):
  assert counts[i]==maxn # These declared fixtures have no unused neighbor padding.
  row=abs(potential[i]);H[i,i]=complex(potential[i])
  for k in range(counts[i]):
   at=i*maxn+k;j=neighbors[at];assert 0<=j<n;hop=complex(real[at],imag[at]);H[i,j]=H.get((i,j),0j)+hop;J[i,j]=J.get((i,j),0j)-1j*hop*dx[at];row+=abs(hop)
  normbound=max(normbound,row)
 herm=max(abs(z-H.get((j,i),0j).conjugate())for(i,j),z in H.items());current=max(abs(z-J.get((j,i),0j).conjugate())for(i,j),z in J.items());assert herm==current==0;assert normbound<emax and max(abs(e)for e in energy)<emax;assert nm>=2 and all(t>0 for t in time)
 norms=[math.fsum(a*a+b*b for a,b in zip(re,im))for re,im in states];expected=n/2 if spin else n;assert all(abs(z-expected)<1e-10+1e-12*expected for z in norms)
 if spin:
  assert n%2==0;assert all(re[i]==im[i]==0 for re,im in states for i in range(1,n,2))
 records.append({'case':name,'n':n,'neighbor_count':maxn,'moments':nm,'energies':ne,'times':nt,'random_vectors':rv,'local_orbitals':local.tolist(),'flags':{'vac0':bool(v0),'vac':bool(vac),'msd':bool(msd),'spin':bool(spin),'ldos':bool(ldos)},'volume':volume,'energy_max':emax,'Gershgorin_absolute_row_bound':normbound,'Hamiltonian_Hermitian_max_error':herm,'wrapped_bond_current_Hermitian_max_error':current,'actual_random_state_norm_squared':norms,'source_expected_random_norm_squared':expected,'fixture_bytes':file.stat().st_size,'fixture_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
(p/'problem_input_audit.json').write_text(json.dumps({'pass':True,'source_problem_only_not_GPU_acceptance':True,'cases':records,'scientific_limits':'Gershgorin scaling andHermiticity are actual declaredproblem checks, not anexactDOS ortransportconvergencecertificate. Seededsourcedefined states verified; no source outputs used as targetcomputation.'},indent=2)+'\n');print([(r['case'],r['n'],r['Gershgorin_absolute_row_bound'])for r in records])
