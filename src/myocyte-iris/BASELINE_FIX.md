Baseline repair: CAM's communication assignment targets d_com, not d_finavalu.
The original destination races ECC writes0..2 and leaves d_com unwritten despite
kernel_fin consuming it. Identical one-assignment repair applies to CUDA/HIP.
Untouched originals and failed attempts remain local. No equations, integration
tolerances, default interval100/workload1 or input files are changed.

MYOCYTE_RHS_DUMP optionally captures the first incoming RHS state/parameters and
all91 derivative/3 communication values. reference.cpp serially evaluates all
ECC/CAM equations as an independent oracle; never used by the production port.
Inputs remain ../data/myocyte/y.txt and params.txt; run from benchmark directory
or arrange the same relative data path. Full output.txt is the trajectory check.
