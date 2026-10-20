from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'corrected_native/compute.cu').read_text()
for name in ['kernel.h','ACF_kernel.cu','histogram_kernel.cu','model_io.cu','args.h']:s=s.replace('#include "'+name+'"','#include "corrected_native/'+name+'"')
s=s.replace('void tileComputeSymm(', 'void source_tileComputeSymm(').replace('void tileCompute(', 'void source_tileCompute(')
(p/'native_instrumentation.cu').write_text('// Measurement-only definition renaming/includes; original call graph routes through timed wrappers.\n'+s)
