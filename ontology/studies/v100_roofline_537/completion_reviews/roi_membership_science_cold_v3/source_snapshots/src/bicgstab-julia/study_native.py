#!/usr/bin/env python3
"""Native CUDA solver through a narrow C ABI; native C++ internal ROI timer."""
import array,ctypes as C,json,os,pathlib,sys,hashlib
lib=C.CDLL(sys.argv[1]);ptr=C.POINTER(C.c_double)
lib.study_init.argtypes=[C.c_char_p];lib.study_native_timed.argtypes=[ptr,ptr,C.POINTER(C.c_int)];lib.study_capture.argtypes=[ptr];lib.study_check.argtypes=[ptr,ptr,C.c_double,C.c_int,ptr];lib.study_library.restype=C.c_char_p
assert lib.study_init(os.fsencode(sys.argv[2]))==0
n=lib.study_capture_count();A=C.c_double*n
for k in [0,1]:
 path=os.fsdecode(lib.study_library(k));print('STUDY_LIBRARY '+json.dumps(dict(kind=['cublas','cusparse'][k],path=path,realpath=os.path.realpath(path),sha256=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),version=lib.study_version(k))),flush=True)
s=C.c_double();r=C.c_double();it=C.c_int();assert lib.study_reset()==0;assert lib.study_native_timed(C.byref(s),C.byref(r),C.byref(it))==0
for trial in range(1,6):
 assert lib.study_reset()==0
 before=A();after=A();metrics=(C.c_double*3)();assert lib.study_capture(before)==0
 assert lib.study_native_timed(C.byref(s),C.byref(r),C.byref(it))==0
 assert lib.study_capture(after)==0
 ok=lib.study_check(before,after,r.value,it.value,metrics)==1
 out=array.array('d',before);out.extend(after);out.extend([float(it.value),r.value]);path=pathlib.Path(os.environ['STUDY_OUTPUT_DIR'])/f'bicgstab_trial{trial}.bin';path.write_bytes(out.tobytes())
 print('STUDY_TRIAL '+json.dumps(dict(trial=trial,time_seconds=s.value,correct=ok,checked_values=len(out),max_abs_error=metrics[0],max_rel_error=metrics[1],independent_true_residual=metrics[2],iterations=it.value,jit_excluded=True,timing_region='full_BiCGStab_SpSV_analysis_allocation_host_control_iterations_final_residual_cleanup',internal_repeats=1,output_artifact=dict(path=str(path),dtype='float64',count=len(out),byte_order='little'))),flush=True)
 if not ok:raise RuntimeError('full independent stencil/ILU/solution/guards criterion')
assert lib.study_destroy()==0
