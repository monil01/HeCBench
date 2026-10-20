#!/usr/bin/env python3
from pathlib import Path
import re,json,hashlib,shutil
p=Path(__file__).resolve().parent;s=p/'corrected_native';d=p/'instrumented_native';shutil.copytree(s,d,dirs_exist_ok=True)
f=d/'src/sgd_k128_kernel_hogwild_warp32.h';text=f.read_text();assert text.count('    int v_id)')==1;text=text.replace('    int v_id)','    int v_id, long long* study_trace)');needle='      start_id = __shfl_sync(0xffffffff, start_id, 0);';assert text.count(needle)==1;text=text.replace(needle,needle+'\n      if(lane_id==0 && study_trace) study_trace[((ite-current_iter)*update_count_this_block+update_ite)*(gridDim.x*4)+wid]=start_id;');f.write_text(text)
f=d/'src/sgd.cu';text=f.read_text();pattern=r'(sgd_k128_kernel_hogwild_warp32_lrate\s*<<<.*?>>>\s*\()(.*?)(\);)';count=[0]
def patch(m):
 count[0]+=1;return 'mfobs::before_training();\n    '+m.group(1)+m.group(2)+',study_trace_ptr'+m.group(3)
text=re.sub(pattern,patch,text,flags=re.S);assert count[0]==3,count
text=text.replace('random_init<<<blockNum, blockSize>>>','mfobs::before_feature(state_size,seg*k,k,scale,i);\n    random_init<<<blockNum, blockSize>>>')
codes={'init_rand_state':0,'sgd_k128_kernel_hogwild_warp32_lrate':1,'transform_half':2,'random_init':3}
for name,code in codes.items():text=re.sub(r'\b'+name+r'(?=\s*<<<)',f'mfobs::launch({code});\n  '+name,text)
text='#include "../../study_observer.h"\n#define cudaMalloc mfobs::allocate\n#define cudaFree mfobs::release\n'+text+'\n#undef cudaMalloc\n#undef cudaFree\n';f.write_text(text)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();review={'original_sized_rng_correction_separate':str(p/'reference_correction_review.json'),'instrumentation':'Noorderingchangeorfactorarithmeticchange; readonlyunique perwarp/perdraw selectortrace store; host launchcounters/actualinputreadbacks/guardalloc/lifetimecaptures. Beforetrainingcompletionfence+actualsnapshot is an explicitmeasurementcovariate, primaryonepartitiononly.','traced_call_sites':count[0],'source_kernel_body_added_line':'if(lane_id==0 && study_trace) study_trace[((ite-current_iter)*update_count_this_block+update_ite)*(gridDim.x*4)+wid]=start_id;','arithmetic_body_unchanged_after_stripping_trace':True,'file_hashes':{str(f.relative_to(p)):sha(f)for f in d.rglob('*')if f.is_file()}}
a=(s/'src/sgd_k128_kernel_hogwild_warp32.h').read_text();b=(d/'src/sgd_k128_kernel_hogwild_warp32.h').read_text().replace('    int v_id, long long* study_trace)','    int v_id)').replace('\n      '+review['source_kernel_body_added_line'],'');assert a==b;(p/'native_instrumentation_review.json').write_text(json.dumps(review,indent=2)+'\n');print('Instrumented3calls, sourceSGDbodyidentical after documentedobserverstripping')
