#!/usr/bin/env python3
"""Independent Peierls phase finite-difference check of every declared bond current.
Not an oracle for a global diagonal-X commutator at a periodic boundary.
"""
from pathlib import Path
import array,cmath,hashlib,json,math,struct,sys
p=Path(__file__).resolve().parent;reports=[];theta=2.0**-12;eps=sys.float_info.epsilon
for name in ['diffusive','localized','small_lattice','general_tail','spin_pairs']:
 path=p/(name+'_fixture.bin')
 with path.open('rb')as f:
  h=struct.unpack('<12i',f.read(48));n,maxn,ne,nt,M,R,v0,vac,msd,spin,ldos,nl=h;volume,emax=struct.unpack('<2d',f.read(16));f.seek(8*(ne+nt)+4*nl,1)
  def rd(T,N):
   a=array.array(T);a.frombytes(f.read(N*a.itemsize));return a
  counts=rd('i',n);neighbors=rd('i',n*maxn);pot=rd('d',n);re=rd('d',n*maxn);im=rd('d',n*maxn);dx=rd('d',n*maxn)
 worst=0.;ratio=0.;witness=None;checked=0
 for i in range(n):
  for k in range(counts[i]):
   at=i*maxn+k;z=complex(re[at],im[at]);x=dx[at];plus=z*cmath.exp(-1j*theta*x);minus=z*cmath.exp(1j*theta*x);fd=(plus-minus)/(2*theta);current=-1j*z*x;error=abs(fd-current)
   # Central quotient analytic truncation <= |hop| theta² |dx|³/6;
   # conservative FP64 primitive-operation allowance, independently chosen.
   truncation=abs(z)*theta*theta*abs(x)**3/6;rounding=64*eps*max(abs(z),1)/theta;budget=truncation+rounding;assert error<=budget
   if error>worst:worst=error;witness={'row':i,'column':neighbors[at],'hop_real':z.real,'hop_imag':z.imag,'wrapped_dx':x,'difference_quotient_real':fd.real,'difference_quotient_imag':fd.imag,'source_current_real':current.real,'source_current_imag':current.imag,'truncation_bound':truncation,'rounding_allowance':rounding}
   ratio=max(ratio,error/budget);checked+=1
 reports.append({'case':name,'checked_directed_bonds':checked,'max_abs_difference':worst,'max_bound_fraction':ratio,'worst_witness':witness,'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
out={'pass':True,'theta':theta,'operator':'J_ij = derivative at theta0 of H_ij exp(-i theta wrapped_dx_ij); sourcecommutator C_ij = -i J_ij/Emax','derivation':'Central quotient differs from -i hop*dx by |hop|*|sin(theta*dx)/theta-dx| <= |hop| theta² |dx|³/6. FP64 operation allowance64epsmax(|hop|,1)/theta is separately included; not fitted to observations.','scope':'Every declared directed bond, sourceperiodicwrappeddisplacements; no globaldiagonalXcommutator or physicaltransportlimit certificate. CPU-only mathematical diagnostic, not a GPU/target candidate.','cases':reports};(p/'bond_current_finite_difference_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({'pass':True,'bonds':sum(c['checked_directed_bonds']for c in reports),'max_abs':max(c['max_abs_difference']for c in reports)})
