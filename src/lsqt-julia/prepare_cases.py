#!/usr/bin/env python3
"""Source-faithful bounded original inputs plus declared small mathematical controls."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;original=p/'dataset_readiness/extracted/examples/fan2018cpc/lattice';out=p/'cases';out.mkdir(exist_ok=True)
manifest=[]
for name,dims in [('diffusive',(128,128,1)),('localized',(16384,1,1))]:
 d=out/name;d.mkdir(exist_ok=True);files=[]
 for n in ['para.in','energy.in','time_step.in','lattice.in']:
  src=original/name/n;text=src.read_text()
  if n=='lattice.in':lines=text.splitlines();lines[0]=' '.join(map(str,dims));text='\n'.join(lines)+'\n'
  (d/n).write_text(text);files.append({'path':str((d/n).relative_to(p)),'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'sha256':hashlib.sha256((d/n).read_bytes()).hexdigest(),'change':'lattice dimensions first line only'if n=='lattice.in'else'none'})
 manifest.append({'name':name,'kind':'original_lattice_solver_parameters_bounded_size','n':dims[0]*dims[1]*dims[2],'files':files})
def energy(d,count,limit):
 (d/'energy.in').write_text(str(count)+'\n'+'\n'.join(format(-limit+2*limit*i/(count-1),'.17g')for i in range(count))+'\n')
def steps(d): (d/'time_step.in').write_text('3\n0.2\n0.125\n0.3\n')
d=out/'small_lattice';d.mkdir(exist_ok=True)
lines=(original/'diffusive/lattice.in').read_text().splitlines();lines[0]='4 3 1';(d/'lattice.in').write_text('\n'.join(lines)+'\n')
(d/'para.in').write_text('model 1\nanderson_disorder 4\nenergy_max 6.1\nnumber_of_moments 32\ncalculate_vac0\ncalculate_vac\ncalculate_msd\ncalculate_ldos\n')
(d/'local_orbitals.in').write_text('3\n0\n5\n11\n');energy(d,17,5.5);steps(d)
manifest.append({'name':'small_lattice','kind':'source_lattice_12site_disorder_plus_all_transport_and_LDOS_controls','n':12})
for name,N,spin in [('general_tail',257,False),('spin_pairs',18,True)]:
 d=out/name;d.mkdir(exist_ok=True);M=1 if spin else 2
 (d/'para.in').write_text('model 0\nenergy_max 1.5\nnumber_of_moments 32\n'+('calculate_spin\n'if spin else'calculate_vac0\ncalculate_vac\ncalculate_msd\n')+'calculate_ldos\n')
 rows=[];hops=[]
 for i in range(N):
  if spin:
   rows.append('1 '+str(i^1));hops.append('0.25 '+('0.5'if i%2==0 else'-0.5'))
  else:
   rows.append('2 '+str((i-1)%N)+' '+str((i+1)%N));hops.extend(['-0.5 -0.125','-0.5 0.125'])
 (d/'neighbor.in').write_text(f'{N} {M}\n'+'\n'.join(rows)+'\n')
 (d/'hopping.in').write_text('complex\n'+'\n'.join(hops)+'\n')
 (d/'position.in').write_text(f'{N} {N}\n'+'\n'.join(str(i//2 if spin else i)for i in range(N))+'\n')
 (d/'potential.in').write_text('\n'.join(str(.125 if i%2==0 else-.125)for i in range(N))+'\n')
 (d/'local_orbitals.in').write_text('3\n0\n1\n'+str(N-1)+'\n');energy(d,19,1.3);steps(d)
 manifest.append({'name':name,'kind':'general_defined_Hermitian_complex_hopping_control','n':N,'spin':spin,'no_neighbor_padding':True})
for c in manifest:c['input_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in(out/c['name']).iterdir()if f.is_file()}
(p/'case_manifest.json').write_text(json.dumps({'cases':manifest,'source_original_unchanged':True,'original4million_sizes_unmeasured':True},indent=2)+'\n')
print([(c['name'],c['n'])for c in manifest])
