# Finite-candidate Michalewicz objective and seeded GPU minimum

All four versions execute the original Float32 objective and GPU atomic
minimum on source-legal inputs in[0,4], dimensions2/5/10 and257/4099points.
The incumbent starts at+0 once percase;100kernel repeats retain it, exactly
as source. The operator is min(0, sampled objectives), not necessarily the
minimum over only supplied points and never a global-optimization certificate.
The source's printed error versus a known global minimum describes search
quality rather than numerical fidelity. The usual test-function domain is
[0,pi] (https://www.sfu.ca/~ssurjano/michal.html); source[0,4] is retained.

Two fixtures include varied full points/zero/pi/4 witnesses and all4positive-
objective batches. Full174648Float32values/capture include148308actual input/
initial-state prefix values,26136actual perpoint objectives,12actualtimed GPU
minima, and guards. Primary minima come from the timed kernel; fullobjective
arrays are separate untimed GPUcaptures, not arrays saved during the timed
atomic reduction. All20captures pass bothreferences and independent Float64
true-pi/sin/fsum checks under predeclared dim-dependent computational budgets.
Every actual input/initialincumbent/guard is exact; positive batches retain
bit-exact+0incumbent. Timed minima are bit-consistent with their captured GPU
objective minima under the same seeded rule. No tolerance was changed.

Historical CUDA.jl genuinely evaluated the objective but reduced on CPU
outside ROI, omitting native atomicwork. It also used minimum(Array) without
including the0seed, despite its CPUchecker including that seed. An actual
untouched historicalGPU diagnostic on257all4points returns positive minima
.171918/.911105/1.319539 for2/5/10dimensions; originalCPUchecker returns0
and wouldFAIL. Raw arrays/source/diagnostic hashes are retained. Current
CUDA.jl aligns GPUminimum and sourceincumbent; originalnativeeval/function/
atomicMin bodies remain unchanged. No native algorithm repair is claimed.

ROI is sum12caseaverage kernel-launch/contextcompletion windows, R100,
reset0 outsideeachwindow and nonebetweenrepeats. Allocation/transfers/JIT/
fullobjectivecapture/readback/CPUchecks are outside. Allfive trials checked,
mean2..5, allconstituenttimes retained. Later internal repeats see a retained
minimum; actual atomic counts were not profiled. This is not a reset-per-
launch contention benchmark, device-only latency, or measured roofline.
Currentauthoredcandidate/deviceprocess1 eachmodel,2kerneldefinitions each;
legacydiagnostic separate, original translation attempts/tokens unknown.
