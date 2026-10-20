#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys,json,hashlib
P=Path(__file__).resolve().parent;R=P.parent.parent;S=R/'src/logic-resim-cuda/Simulation/src';source=S/'parser/interParser.cpp';raw=source.read_text();s=raw.replace('    _simulator->addEvalMgr(h_eTable, EvalC);','    // CPU input decoder: no GPU truth-table upload; no Simulator exists here.').replace('h_eTable = new char[tableSize];','h_eTable = new char[tableSize](); // Defined unused parser scratch, static source hazard recorded.')
assert s!=raw and '_simulator->addEvalMgr(h_eTable, EvalC);'not in s
copy=P/'source_interParser_input_decode.cpp';copy.write_text(s)
cmd=['g++','-O3','-std=c++17','-ffunction-sections','-fdata-sections','-Wl,--gc-sections']+['-I'+str(S/d)for d in ['gate','parser','sim','util','wave']]+[str(P/'prepare_inputs.cpp'),str(copy)]+[str(f)for directory in ['gate','parser','util','wave']for f in sorted((S/directory).glob('*.cpp'))if f.name!='interParser.cpp']+['-o',sys.argv[1]]
subprocess.run(cmd,check=True)
(P/'input_decoder_provenance.json').write_text(json.dumps({'source':str(source.relative_to(R)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'owned_copy':str(copy.relative_to(R)),'owned_copy_sha256':hashlib.sha256(copy.read_bytes()).hexdigest(),'changes':['Skip unusedGPUtruthupload inCPU-onlyparseHeader; nullptrSimulator isneverconstructed/called','Zeroinitializeunusedpackedtruthscratch toavoid sourceUB; parsedCPUwaveformsunchanged'], 'CPU_only':True,'GPU_kernel_invocations':0,'is_benchmark_candidate_or_validation_pass':False,'unreachable_GPU_dispatch_sections_removed_by_linker_gc':True,'command':cmd},indent=2)+'\n')
