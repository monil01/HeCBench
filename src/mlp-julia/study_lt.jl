using CUDA,Libdl,SHA
const CB=CUDA.CUBLAS
function mlp_lt_layer(handle,x,w,bias,y,ifeat,ofeat,use_bias,use_relu,workspace)
 # Active descriptor sizes from the pinned installed CUDA12.0 native headers;
 # generated CUDA6.2 bindings allocate larger opaque storage. Internal Init ABI
 # is exactly the C++ inline wrapper operation, not a heap Create/Destroy substitute.
 desc=Ref(CB.cublasLtMatmulDescOpaque_t(ntuple(_->UInt64(0),32)))
 ad=Ref(CB.cublasLtMatrixLayoutOpaque_t(ntuple(_->UInt64(0),14)));bd=Ref(CB.cublasLtMatrixLayoutOpaque_t(ntuple(_->UInt64(0),14)));cd=Ref(CB.cublasLtMatrixLayoutOpaque_t(ntuple(_->UInt64(0),14)));pref=Ref(CB.cublasLtMatmulPreferenceOpaque_t(ntuple(_->UInt64(0),12)))
 datatype=convert(CB.cudaDataType,Float32);CB.cublasLtMatmulDescInit_internal(desc,184,CB.CUBLAS_COMPUTE_32F,datatype)
 ta=Ref(CB.CUBLAS_OP_T);tb=Ref(CB.CUBLAS_OP_N);CB.cublasLtMatmulDescSetAttribute(desc,CB.CUBLASLT_MATMUL_DESC_TRANSA,ta,4);CB.cublasLtMatmulDescSetAttribute(desc,CB.CUBLASLT_MATMUL_DESC_TRANSB,tb,4)
 if use_bias
  bp=Ref(pointer(bias));CB.cublasLtMatmulDescSetAttribute(desc,CB.CUBLASLT_MATMUL_DESC_BIAS_POINTER,bp,8)
 end
 ep=Ref(use_bias ? (use_relu ? CB.CUBLASLT_EPILOGUE_RELU_BIAS : CB.CUBLASLT_EPILOGUE_BIAS) : (use_relu ? CB.CUBLASLT_EPILOGUE_RELU : CB.CUBLASLT_EPILOGUE_DEFAULT));CB.cublasLtMatmulDescSetAttribute(desc,CB.CUBLASLT_MATMUL_DESC_EPILOGUE,ep,4)
 CB.cublasLtMatrixLayoutInit_internal(ad,64,datatype,ifeat,ofeat,ifeat);CB.cublasLtMatrixLayoutInit_internal(bd,64,datatype,ifeat,128,ifeat);CB.cublasLtMatrixLayoutInit_internal(cd,64,datatype,ofeat,128,ofeat);CB.cublasLtMatmulPreferenceInit_internal(pref,64)
 limit=Ref(Csize_t(1<<22));CB.cublasLtMatmulPreferenceSetAttribute(pref,CB.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,limit,sizeof(Csize_t))
 heuristic=Ref(CB.cublasLtMatmulHeuristicResult_t(CB.cublasLtMatmulAlgo_t(ntuple(_->UInt64(0),8)),Csize_t(0),CB.CUBLAS_STATUS_SUCCESS,0f0,ntuple(_->Int32(0),4)));returned=Ref{Cint}(0);CB.cublasLtMatmulAlgoGetHeuristic(handle,desc,ad,bd,cd,cd,pref,1,heuristic,returned);returned[]>0||error("Ltnoheuristic");algorithm=Ref(heuristic[].algo);alpha=Ref(1f0);beta=Ref(0f0)
 CB.cublasLtMatmul(handle,desc,alpha,w,ad,x,bd,beta,y,cd,y,cd,algorithm,workspace,1<<22,CUDA.default_stream())
 return
end
function mlp_lt_factory(x,weights,biases,outputs,use_bias,use_relu)
 handle=Ref{CB.cublasLtHandle_t}(C_NULL);CB.cublasLtCreate(handle);workspace=CUDA.zeros(UInt8,16*1024*1024)
 println("STUDY_LT_ABI active_desc=184 active_layout=64 active_pref=64 generated_desc=",sizeof(CB.cublasLtMatmulDescOpaque_t)," generated_layout=",sizeof(CB.cublasLtMatrixLayoutOpaque_t)," generated_pref=",sizeof(CB.cublasLtMatmulPreferenceOpaque_t)," version=",CB.cublasLtGetVersion()," allocated_workspace_bytes=",length(workspace)," active_preference_bytes=",1<<22)
 for name in Libdl.dllist()
  if occursin("libcublasLt",name)
   path=realpath(name);println("STUDY_VENDOR_LIBRARY path=",path," sha256=",bytes2hex(sha256(read(path))))
  end
 end
 op=()->begin
  input=x
  for (layer,(ifeat,ofeat))in enumerate(((784,512),(512,512),(512,10)))
   mlp_lt_layer(handle[],input,weights[layer],biases[layer],outputs[layer],ifeat,ofeat,use_bias,use_relu,workspace);input=outputs[layer]
  end
 end
 cleanup=()->CB.cublasLtDestroy(handle[])
 (op,cleanup)
end
