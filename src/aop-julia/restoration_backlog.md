# Open American-option reference restoration

Historical CUDA.jl explicitly implements European terminal-payoff Monte Carlo in place of native American Longstaff–Schwartz continuation regression/early exercise. Preserve both original sources. Mathematical witness: default EuropeanBSput0.38443077916 is below immediateAmericanexercisebound0.4.

Required restoration: all native path states, prepareSVD/QR-Jacobi cutoff, beta partial/final reductions, backward cashflow decisions and finalprice. Preserve PUT/CALL, timestep/grid scopes, samplebytes, cashflow aliasing and sourceROI(samplecopy through scalarreadback). Independent fullpath/least-squares cashflow oracle and rank/threshold/conditioning checks required. Nativebarrier/min-in-the-money/partialtile behavior must be reviewed; no specific source runtimefailure claimed yet. Exact original native instructions/libversions and actualGPUprobe pending. No KA/JACC implementation drafted.

Status reference_restore_pending, repairablebacklog, notterminal/unsupported. Next source restoration may use inherited source audit; record cost separately.
