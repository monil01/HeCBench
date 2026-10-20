#!/usr/bin/env python3
from pathlib import Path
import re
P=Path(__file__).resolve().parent;s=(P/'corrected_numeric.cuh').read_text()
s='#include "study_interface.h"\n#include <chrono>\n'+s
# Each runtime failure is checked; no source math/source allocations removed.
for api in ['cudaMalloc','cudaMemcpy','cudaMemset','cudaDeviceSynchronize','cudaFree']:
 s=re.sub(r'\b'+api+r'\(([^;]*?)\);',lambda m:'SLU_CUDA_CHECK('+api+'('+m[1]+'));',s)
pattern=r'(RL(?:_perturb|_onecol_factorizeCurrentCol(?:_perturb)?|_onecol_updateSubmat|_onecol_cleartmpMem)?)<<<([^>]+)>>>\((.*?)\);'
ops={'RL':0,'RL_perturb':1,'RL_onecol_factorizeCurrentCol':2,'RL_onecol_factorizeCurrentCol_perturb':3,'RL_onecol_updateSubmat':4,'RL_onecol_cleartmpMem':5}
def hook(m):
 name,geometry,args=m[1],m[2],m[3];op=ops[name]
 if op<2:threads='int(dimBlock.x)';blocks='int(dimGrid.x)';smem='int(MemSize)';col='0';head='A_sym.level_ptr[i]';pos='j*TMPMEMNUM';stream='0'
 else:threads='256';blocks='int(subMatSize)'if op==4 else'1';smem='0';col='currentCol';head=pos='0';stream='j'
 return '{ SCall q={'+f'{op},{threads},{blocks},{smem},n,nnz,{col},{head},{pos},{stream},pert,0'+' ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts['+str(op)+']++;if(study_cb){if(study_cb(&q))abort();}else '+m[0]+' SLU_CUDA_CHECK(cudaGetLastError()); }'
s,n=re.subn(pattern,hook,s,flags=re.S);assert n==10,n
marker='  Timer t;\n  double utime;\n  t.start();'
assert s.count(marker)==1
s=s.replace(marker,'  void* study_pointers[]={sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem};\n  study_before(A_sym,study_pointers,tmpMem,PERTURB,pert);\n  SLU_CUDA_CHECK(cudaDeviceSynchronize());\n  auto study_start=std::chrono::steady_clock::now();\n'+marker)
s=s.replace('  t.elapsedUserTime(utime);','  study_roi_seconds(std::chrono::duration<double>(std::chrono::steady_clock::now()-study_start).count());\n  t.elapsedUserTime(utime);')
s=s.replace('  //check NaN elements','  //check NaN elements')
marker='  SLU_CUDA_CHECK(cudaMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost));'
assert s.count(marker)==1;s=s.replace(marker,marker+'\n  study_after(A_sym,study_pointers,tmpMem);')
(P/'integrated_numeric.cuh').write_text(s)
s=(P/'reference_history/src/main.cpp').read_text().replace('int main(int argc, char** argv)','int source_main(int argc, char** argv)')
marker='  vector<REAL> x = A_sym.solve(nicslu, b);';assert s.count(marker)==1;s=s.replace(marker,marker+'\n  study_solved(A_sym,x,matrixName,PERTURB);')
# Repeated inprocess runs replace the original process-exit cleanup, outside ROI.
s=s.replace('  return 0;','  NicsLU_Destroy(nicslu);free(nicslu);free(ax);free(ai);free(ap);\n  return 0;')
(P/'integrated_main.cuh').write_text(s)
