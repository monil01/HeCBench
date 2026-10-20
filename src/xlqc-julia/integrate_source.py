#!/usr/bin/env python3
"""Add named protocol hooks around the unchanged source Hartree–Fock pipeline."""
from pathlib import Path
import re,hashlib,json,difflib
P=Path(__file__).resolve().parent;raw=(P/'reference_history/main.cu').read_text();s=raw.replace('int main(int argc, char* argv[])','int xl_original_main(int argc, char* argv[])')
# Device output guard allocation changes only allocation base/padding; source bodies untouched.
s=s.replace('cudaMalloc(', 'xl_guarded_malloc(').replace('cudaFree(', 'xl_guarded_free(')
needle='    fprintf(stdout, "%5s %20s %20s %20s %20s\\n",'
assert s.count(needle)==1
s=s.replace(needle,'    xl_initial(use_dp,n_pbf,p_basis->num,n_elec,ene_nucl,dev_pbf_xlec,dev_pbf_to_cbf,dev_mat_Q,h_pbf_xlec,h_pbf_to_cbf,h_mat_Q,S,T,V,H_core,S_invsqrt);\n'+needle)
for label,op in [('J',0),('K',1)]:
 pattern=r'        if \(use_dp\) \{\s*cuda_mat_'+label+r'_PI_dp<<<grid_size, block_size>>>\s*\(dev_pbf_xlec, dev_pbf_to_cbf, n_pbf, dev_mat_D, dev_mat_'+label+r'_PI, dev_mat_Q\);\s*\} else \{\s*cuda_mat_'+label+r'_PI<<<grid_size, block_size>>>\s*\(dev_pbf_xlec, dev_pbf_to_cbf, n_pbf, dev_mat_D, dev_mat_'+label+r'_PI, dev_mat_Q\);\s*\}'
 replacement=f'        xl_dispatch({op}+2*use_dp,n_pbf,dev_pbf_xlec,dev_pbf_to_cbf,dev_mat_D,dev_mat_{label}_PI,dev_mat_Q);'
 s,count=re.subn(pattern,replacement,s);assert count==1,(label,count)
 needle=('        auto kstart = std::chrono::steady_clock::now();'if label=='J'else'        kstart = std::chrono::steady_clock::now();')
 assert s.count(needle)==1
 s=s.replace(needle,f'        xl_before(iter,{op}+2*use_dp,dev_mat_D,dev_mat_{label}_PI,h_mat_D);\n'+needle)
 needle=f'        my_cuda_safe(cudaMemcpy(h_mat_{label}_PI, dev_mat_{label}_PI, n_PI_bytes, cudaMemcpyDeviceToHost),"mem_mat_{label}_PI");'
 assert s.count(needle)==1
 s=s.replace(needle,needle+f'\n        xl_after(iter,{op}+2*use_dp,dev_mat_{label}_PI,h_mat_{label}_PI);')
needle='        time_in_usec = ktime * 1e-3f;';assert s.count(needle)==2;s=s.replace(needle,'        xl_elapsed_ns += static_cast<uint64_t>(ktime);\n'+needle)
needle='        fprintf(stdout, "%5d %20.10f", iter, ene_total);';assert s.count(needle)==1;s=s.replace(needle,'        xl_iteration(iter,ene_total,delta_E,rms_D,dd_max,D,Fock,Coef,emo,J,K);\n'+needle)
needle='    fprintf(stdout, "SCF converged! E_total = %20.10f\\n", ene_total);';assert s.count(needle)==1;s=s.replace(needle,'    xl_final(iter,ene_total,D,Fock,Coef,emo);\n'+needle)
(P/'instrumented_main.cuh').write_text(s);(P/'protocol_adapter.patch').write_text(''.join(difflib.unified_diff(raw.splitlines(True),s.splitlines(True),fromfile='original/main.cu',tofile='instrumented_main.cuh')));print('FULL_SOURCE_SCF_HOOKS_GENERATED_NO_GPU_BODY_CHANGES')
