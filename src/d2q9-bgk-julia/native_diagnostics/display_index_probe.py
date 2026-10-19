#!/usr/bin/env python3
from pathlib import Path
import subprocess,os,fcntl,json
P=Path(__file__).resolve().parent;source=P.parent.parent/'d2q9-bgk-cuda/main.cu'
q=subprocess.run(['nvcc','-O1','-g','-std=c++17','-arch=sm_70','-Xcompiler','-fsanitize=address',str(source),'-o',str(P/'original_cli_asan')],capture_output=True,text=True);(P/'asan_build.log').write_text(q.stdout+q.stderr);q.check_returncode()
(P/'asan_outputs').mkdir(exist_ok=True)
for name in ['input.params','obstacles.dat']:(P/'asan_outputs'/name).write_bytes((P/name).read_bytes())
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',ASAN_OPTIONS='detect_leaks=0');v=subprocess.run([str(P/'original_cli_asan'),'input.params','obstacles.dat'],cwd=P/'asan_outputs',env=env,capture_output=True,text=True,timeout=120);(P/'asan_run.log').write_text(v.stdout+v.stderr)
observed='heap-buffer-overflow' in v.stderr and 'write_values' in v.stderr
(P/'display_index_witness.json').write_text(json.dumps({'actual_asan_observed_host_heap_buffer_overflow_in_write_values':observed,'exit_code':v.returncode,'source_expression':'obstacles[ii * params.nx + jj]','correct_row_major_expression':'obstacles[ii + jj * params.nx]','named_shape':[128,8],'maximum_source_index':127*128+7,'allocated_obstacle_count':1024,'scope':'Final text display obstacle flag only; source device kernels and independently checked distribution/reduction arrays unchanged. Original uninstrumented CLI exit0 was readiness, not evidence of valid output flag. No silent native correction.'},indent=2)+'\n')
print(json.dumps({'actual_host_overflow_observed':observed,'exit_code':v.returncode}))
