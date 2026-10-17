
## Completion/ordering measurement correction

RawcuBLAS defaultlegacy0 handle completion is now device-wide with CUDA.device_synchronize(blocking=true). Native unchanged five captures reused; allthreeJulia five fresh. Oldnumericalcaptures retained, oldtimingexcluded before replacement. No observed numerical failure or quantified timing bias inferred from missing ordering/completion proof. Not a translation or tuning attempt.

Corrected configuration `MoE_FourExperts_K32N48_tokens32_nativeRouting_FP16_groupedEx129_v1__devicewide_completion_v2`, result `.porting-state/portable_julia/runs/20261017T224954119665Z_blas-groupgemm/result.json`. All five correctness checks against both references pass. Time1 alone is discarded; mean times2–5 in seconds: cuda=2.32768975e-05, cuda_jl=2.3299485000000002e-05, ka=6.5215925e-06, jacc=8.96703e-06. Original translation attempts/tokens and commit receipts remain distinct from this shared measurement repair.
