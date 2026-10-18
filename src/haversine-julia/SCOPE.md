# Spherical great-circle distances

All four versions pass five complete Float64 captures on4099 ordinary/city
pairs (1mm computational sphere-distance budget) and99 exact/near-antipodal
pairs (1m budget predeclared for ill conditioning). The independent Python
oracle uses 3D unit vectors, cross norm, dot and atan2; device code uses
haversine/asin. R=6371km is the original fixed-radius sphere, not an Earth
ellipsoid or physical accuracy claim. The original huge cities CLI is not
certified. All actual input bits and10 guards are audited per artifact.

Native distance.cu/.h remain unchanged. Historical CUDA.jl is a genuine
SoA implementation and is archived. The study preserves the native macro's
two-operation degree-to-radian order (lat*pi then /180). Native double4 AoS
versus Julia four SoA arrays is an explicit mapping/performance covariate.
Near-antipodal NaN was investigated but NOT observed on the99 named source
pairs. No native fix or clamp was applied. This is not a global proof that
asin's argument always stays in range. Float64 libm oracle is not MPFR.

Each model uses256 threads, one independent pair per worker, no authored
shared memory. ROI is100 repeated kernel launches plus context completion,
reported per launch; allocation/reset/readback/oracle/artifact writes and
JIT are outside. All five runs are checked; mean uses runs2..5. These are
small named workloads, with launch overhead and layout/compiler covariates;
no measured roofline position or universal abstraction ranking is claimed.
