#!/usr/bin/env python3
from pathlib import Path
import re,difflib,json,hashlib
P=Path(__file__).resolve().parent;old=(P/'reference_history/main.cu').read_text();s=old.replace('int main(int argc, char* argv[])','int source_main(int argc, char* argv[])');assert s!=old
s=s.replace('memcpy((result + count),','memmove((result + count),')
# Source GPU body files are included byte-identical; runtime guards/status checks
# and callback selection are host-side study adapters only.
s=s.replace('#include "kernels.cu"','#include "reference_history/kernels.cu"')
s=re.sub(r'cudaMalloc\(\(void\*\*\)&([A-Za-z0-9_]+),\s*([^;]+)\);',r'study_allocate((void**)&\1,\2);',s)
s=re.sub(r'cudaFree\(([^;]+)\);',r'study_free(\1);',s)
for name in ['cudaMemcpy','cudaDeviceSynchronize']:
 s=re.sub(r'\b'+name+r'\(([^;]*?)\);',lambda m:'SS_CUDA_CHECK('+name+'('+m[1]+'));',s)
marker='  double time = 0.0;';assert s.count(marker)==1;s=s.replace(marker,marker+'\n  study_before(textBuf,textLength,subStrBuf,subStrLength,resultBuf,resultCountBuf,workGroupCount,totalSearchPos,searchLenPerWG,text,ss);')
pattern=r'(StringSearchNaive|StringSearchLoadBalance)<<<grid, block, subStrLength>>>\((.*?)\);'
def launch(m):
 op=0 if m[1]=='StringSearchNaive'else 1
 return '{SSCall q={'+f'{op},int(block.x),int(grid.x),int(subStrLength),textLength,subStrLength,searchLenPerWG,0' +',{textBuf,subStrBuf,resultBuf,resultCountBuf}};study_counts['+str(op)+']++;if(study_callback){if(study_callback(&q))throw std::runtime_error("authoredframeworkcallbackfailed");}else '+m[0]+' SS_CUDA_CHECK(cudaGetLastError());}'
s,n=re.subn(pattern,launch,s,flags=re.S);assert n==2,n
marker='    auto end = std::chrono::steady_clock::now();';assert s.count(marker)==2;s=s.replace(marker,marker+'\n    study_roi_seconds(std::chrono::duration<double>(end-start).count());')
marker='    verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);';assert s.count(marker)==2;s=s.replace(marker,'    study_device_outputs();\n    int study_verify_status=verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);\n    study_source_final(resultCount,workGroupCount,result,cpuResults,study_verify_status);')
(P/'configured_main.cuh').write_text(s);(P/'host_adapter.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='reference_history/main.cu',tofile='configured_main.cuh')))
print('sourceGPUbodiesunchanged',hashlib.sha256((P/'reference_history/kernels.cu').read_bytes()).hexdigest())
