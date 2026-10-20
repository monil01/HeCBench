#!/usr/bin/env python3
from pathlib import Path
import re,shutil,json,hashlib
p=Path(__file__).resolve().parent;o=p/'original_native';d=p/'instrumented_native';d.mkdir(exist_ok=True)
for f in o.iterdir():
 if f.is_file():shutil.copy2(f,d/f.name)
names=['gpu_set_zero','gpu_copy_state','gpu_add_state','gpu_apply_sz','gpu_find_inner_product_1','gpu_find_inner_product_2','gpu_apply_hamiltonian','gpu_apply_commutator','gpu_apply_current','gpu_chebyshev_01','gpu_chebyshev_2','gpu_chebyshev_1x','gpu_chebyshev_2x','gpu_kernel_polynomial'];counts={}
for name in ['vector.cu','hamiltonian.cu','sigma.cu']:
 text=(d/name).read_text();text='#include "../study_capture.h"\n'+text
 if name in ['vector.cu','hamiltonian.cu']:
  def hook(m):
   k=m.group(1);counts[k]=counts.get(k,0)+1;return 'lsqtstudy::launch('+str(names.index(k))+');\n  '+m.group(0)
  text=re.sub(r'\b(gpu_\w+)\s*<<<',hook,text)
 if name=='vector.cu':
  before='  CHECK(cudaMemcpy(target_imag, imag_part, array_size, cudaMemcpyDeviceToHost));';assert text.count(before)==1;text=text.replace(before,before+'\n  lsqtstudy::moments(n,target_real,target_imag);')
 if name=='sigma.cu':
  before='    correlation_function[step1] = temp / model.energy_max;\n  }';assert text.count(before)==1;text=text.replace(before,before+'\n  lsqtstudy::scientific(model.number_of_energy_points,correlation_function);')
  before='for (int m = 0; m < model.number_of_steps_correlation; ++m) {';assert text.count(before)==3;text=text.replace(before,before+'\n    lsqtstudy::set_step(m);')
  for function,next_function,lines in [('find_vac','find_msd',['lsqtstudy::final_state(10,model.number_of_atoms,state_left.real_part,state_left.imag_part);','lsqtstudy::final_state(11,model.number_of_atoms,state_right.real_part,state_right.imag_part);']),('find_msd','find_spin_polarization',['lsqtstudy::final_state(12,model.number_of_atoms,state.real_part,state.imag_part);','lsqtstudy::final_state(13,model.number_of_atoms,state_x.real_part,state_x.imag_part);']),('find_spin_polarization',None,['lsqtstudy::final_state(14,model.number_of_atoms,state.real_part,state.imag_part);'])]:
   start=text.index('void '+function+'(');end=text.index('void '+next_function+'(',start)if next_function else len(text);part=text[start:end];at=part.index('  output.close();');part=part[:at]+'  '+'\n  '.join(lines)+'\n'+part[at:];text=text[:start]+part+text[end:]
 if name=='hamiltonian.cu':
  for kernel,values in [('gpu_chebyshev_01','bessel_0,bessel_1,double(direction)'),('gpu_chebyshev_2','bessel_m,double(label)'),('gpu_chebyshev_1x','bessel_1'),('gpu_chebyshev_2x','bessel_m,double(label)')]:
   code=names.index(kernel);needle='lsqtstudy::launch('+str(code)+');';assert text.count(needle)==1;text=text.replace(needle,'lsqtstudy::scalar('+str(code)+',{'+values+'});\n  '+needle)
 (d/name).write_text(text)
assert len(counts)==14
# Confirm original GPU device bodies themselves are byte-identical.
checks=[]
for name in ['vector.cu','hamiltonian.cu']:
 for text in [(o/name).read_text(),(d/name).read_text()]:
  pass
 def bodies(text):
  result={}
  for m in re.finditer(r'__global__\s+void\s+(gpu_\w+)\s*\(',text):
   at=text.index('{',m.end());depth=1;end=at+1
   while depth:depth+= (text[end]=='{')-(text[end]=='}');end+=1
   result[m.group(1)]=text[at:end]
  return result
 a,b=bodies((o/name).read_text()),bodies((d/name).read_text());assert a==b
 checks.extend([{'entry':k,'body_sha256':hashlib.sha256(v.encode()).hexdigest(),'unchanged':True}for k,v in a.items()])
(p/'native_instrumentation_review.json').write_text(json.dumps({'all14GPUbodies_unchanged':True,'checks':checks,'launch_callsite_counts':counts,'host_changes':'Only source observer hooks after originalfullDtoHmoments,postscientificsummation,sourcefinalstates,loopworkmetadata,andbufferlifecycle; enlargedguards/poison explicitlyinstrumented, no scientificnativecorrection.'},indent=2)+'\n')
print({'unchanged_device_entries':len(checks),'observed_launch_callsites':sum(counts.values())})
