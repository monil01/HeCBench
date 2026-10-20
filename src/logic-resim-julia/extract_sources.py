#!/usr/bin/env python3
"""Exact unchanged native GPU body and original CPU fallback methods, source-bound."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;R=P.parent.parent;s=R/'src/logic-resim-cuda/Simulation/src/sim/Simulator.cu';raw=s.read_text();a=raw.index('__global__ void simulateParallel');b=raw.index('\nvoid Simulator::simulateBlock(tHistory',a);gpu=raw[a:b]
a=raw.index('void Simulator::simulate(Gate *g)');cpu=raw[a:]
(P/'source_kernel.cuh').write_text(gpu)
(P/'source_cpu_fallback.cpp').write_text('#include "Simulator.h"\n#include <climits>\n#include <thread>\nusing std::thread;\n// CPU-only object: original CPU methods use dumpOff, not GPU members.\nSimulator::Simulator(tUnit off):dumpOff(off){}\n'+cpu)
(P/'source_extraction.json').write_text(json.dumps({'source':str(s.relative_to(R)),'source_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'GPU_body_sha256':hashlib.sha256(gpu.encode()).hexdigest(),'CPU_methods_sha256':hashlib.sha256(cpu.encode()).hexdigest(),'GPU_body_unchanged':True,'CPU_methods_unchanged':True,'CPU_only_constructor':'InitializesdumpOff, GPUmembersnotusedbyoriginalCPUmethods; no GPU operation substituted.','CPU_fallback_retained_original_Cpp_not_translated':True},indent=2)+'\n');print('Extracted unchanged nativeGPUkernel/originalCPUfallback')
