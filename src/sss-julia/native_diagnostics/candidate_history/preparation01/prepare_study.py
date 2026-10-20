#!/usr/bin/env python3
"""Owned source-host integration, original scientific arithmetic/GPU bodies retained."""
import pathlib,hashlib,json,difflib
P=pathlib.Path(__file__).resolve().parent;O=P/'original_native';C=P/'configured';C.mkdir(exist_ok=True);rows=[]
for f in O.iterdir():
 if not f.is_file()or f.suffix not in ['.cu','.cpp','.h']:continue
 old=f.read_text();s=old
 if f.name=='DPmixGGM_SSSmoves.cpp':
  s=s.replace('#include "kernels.cu"','#include "../study_interface.h"')
  a='''  CanDeleteEdge<<<device.n_delete, BLOCK_SIZE, 0, device.delete_stream>>>(
      device.d_in_delete, device.d_which_delete);''';b='''  sss_launch(0,device.n_delete,0,device.delete_stream,device.d_in_delete,nullptr,device.d_which_delete,size_temp/sizeof(myInt),0);''';assert s.count(a)==1;s=s.replace(a,b)
  a='''  CanAddEdge<<<device.n_add, BLOCK_SIZE, buffsiz, device.add_stream>>>(
      device.d_in_delete, device.d_in_add, device.d_which_add);''';b='''  sss_launch(1,device.n_add,buffsiz,device.add_stream,device.d_in_delete,device.d_in_add,device.d_which_add,0,size_temp/sizeof(myInt));''';assert s.count(a)==1;s=s.replace(a,b)
  # Define only unused copied tail fields, preserving all active entries.
  a='G->Cliques[j][k]';assert s.count(a)==1;s=s.replace(a,'(k < G->CliquesDimens[j] ? G->Cliques[j][k] : 0)')
  a='G->Separators[j][k]';assert s.count(a)==1;s=s.replace(a,'(k < G->SeparatorsDimens[j] ? G->Separators[j][k] : 0)')
  # The no-change draw never uses which_change. Avoid its undefined dead read.
  a='int which_change = which[i];';assert s.count(a)==1;s=s.replace(a,'int which_change = (i < ndecomp ? which[i] : -1);')
  a='  // Make CanDeleteEdge buffer and transfer to the GPUs';assert s.count(a)==1;s=s.replace(a,'  sss_observe_graph(G);\n'+a)
  a='  cudaStreamSynchronize(device.delete_stream);';assert s.count(a)==1;s=s.replace(a,a+'\n  sss_observe_results(0,device.n_delete,device.which_delete);')
  a='  cudaStreamSynchronize(device.add_stream);';assert s.count(a)==1;s=s.replace(a,a+'\n  sss_observe_results(1,device.n_add,device.which_add);')
 if f.name=='main.cu':
  s='#include "../study_interface.h"\n'+s
  a='  data.close();';assert s.count(a)==1;s=s.replace(a,a+'\n  sss_observe_data(X,n,p);')
  a='  // start the stopwatch\n  auto start = std::chrono::steady_clock::now();';assert s.count(a)==1;s=s.replace(a,a+'\n  sss_start_timer();')
  a='    num_allModels += num_cases;';assert s.count(a)==1;s=s.replace(a,a+'\n    sss_observe_iteration(state,k,score,localBestScore,globalBestScore,nmodes,num_cases,num_allModels);')
  a='  auto now = std::chrono::steady_clock::now();';assert s.count(a)==1;s=s.replace(a,'  sss_stop_timer();\n'+a)
  a='  // writing the lists';assert s.count(a)==1;s=s.replace(a,'  sss_observe_final(state,localBestState,globalBestState,modesList,featureList,k,num_allModels);\n'+a)
  # Returning from the renamed main must preserve the implicit original main0.
  i=s.rfind('}');s=s[:i]+'  return 0;\n'+s[i:]
  s=s.replace('cudaMalloc(', 'sss_malloc(').replace('cudaFree(', 'sss_free(')
 (C/f.name).write_text(s)
 if s!=old:(P/'native_diagnostics'/('integration_'+f.name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='original/'+f.name,tofile='configured/'+f.name)))
 rows.append(dict(file=f.name,original_sha256=hashlib.sha256(old.encode()).hexdigest(),configured_sha256=hashlib.sha256(s.encode()).hexdigest(),unchanged=s==old))
(P/'native_diagnostics/integration_source_manifest.json').write_text(json.dumps(dict(benchmark='sss',rows=rows,original_GPU_kernels_unchanged=True,source_CPU_scientific_equations_unchanged=True,qualified_host_definedness=['Unusedcopiedclique/separatortailszero','No-changewhichdeadreadguard','Renamedmainexplicitreturn0'],observations=['Actualinputreadback/allgatepredicates','Fullsearchstatetrace/finalmodels/guards'],pure_Julia_CPU_translation=False),indent=2)+'\n');print('Source-host adapter prepared; original archive unchanged')
