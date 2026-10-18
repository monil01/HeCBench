## Completed result

All four versions passed exact full-artifact verification in `.porting-state/portable_julia/runs/20261018T104646539975Z_spmm/result.json`; source/KA/JACC committed separately through `c3c41f6ae`. See `study_notes.json`, `output_audit.json`, `SCOPE.md` and per-version `verification.json`. The following records the earlier preparation state, not pending work.

# Historical preparation plan: spmm

Source reviewed but NO implementation edits or archives yet. CUDA source actually uses sparse×sparse cusparseSpGEMM, unlike benchmark spgemm (sparse×dense SpMM, owned portable_infrastructure). Historical Julia genuine custom CSR×CSR into DENSE C; restore actual vendor sparse representation and source ROI. Native has zero authored GPU kernels. Preserve full NVIDIA license notice from main.cu in new driver/comments/docs.

Pinned system DSO /usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140, actual API version12001. Follow root spgeam/spmv exact dlsym/dladdr/DSO SHA/owned handle policy. CUDAjl/KA/JACC libraryinterop, no authored GPU lowering or device JIT claim.

Native phase order: allocate/upload A/B Int32 CSR +Coffsets; CreateCSR A/B and C with nnz0/allNULLptr; SpGEMM_createDescr; workEstimation query/allocation/actual outside ROI; compute query and buffer allocation outside ROI; R repeats SpGEMM_compute +devicecompletion timed; GetSize and allocate Cvalues/columns/setCSR pointers/copy outside ROI. Source DEFAULT/F32/NOTRANSPOSE/alpha1beta0. Fresh descriptor/workspaces EACH trial outside ROI because copy changes phase/state; full separate warm pipeline per shape outside all5 timing windows. Preserve repeated compute count as source (named R5 or10 vs1000 unmeasured).

Suggested cases: source1024x1024x1024/nnz1024-permatrix via named signed-permutation/diagonal sparse inputs; plus1x1 and rectangular17x11x23,33x17x19,64x33x65 with emptyrows/cols and multiple productpaths. Bounded dyadic A=s_row*positivequarter, B=s_col*positiveeighth ensures product signs fixed peroutput/no cancellation and all partials exactly representable at1/32 scale. Independent Fraction/dense matmul oracle; sortedunique full CSR structure +everyvalue+allinputs+guards. Avoid wholezeroC initially; no arbitraryfloat conditioning/solverclaim. Original4096/full1000repeats and randominitializer unmeasured.

UInt32rawbits artifact forF32values+I32indices; prefix actualA/B sixarrays and initialized Coffsets only (Cvalues/columns allocatedaftertimedcompute as source). Post includes actual A/B read-only andfullC threevectors/guards afteruntimedcopy. Determine/assert actual NNZ vsoracle beforealloc/copy safely; preserve failedwitness if structuralzeros differ, do not silentlyprune. No source host malloc(sizeofsize_t) rowpadding correction claim: its meaningful payload isInt32 and GPUcopy32bit.

Open prospective windows: spmm/shared_benchmark_workflow andcuda_jl_reference_preparation begun2026-10-18T10:37:53.451Z. No KA/JACC meter yet; open model window BEFORE writing each wrapper if feasible. Root alone Git; queued source commit session8593 pathfinder, poll before any mutations.
