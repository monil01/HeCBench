from pathlib import Path
import shutil
r=Path('/home/imo/HeCBench_julia');d=r/'.porting-state/diagnostics/logic-resim';src=d/'source';dst=d/'metadata_source'
if dst.exists():raise SystemExit('alreadyexists')
shutil.copytree(src,dst)
p=dst/'Simulation/src/sim/Simulator.cu';s=p.read_text();s=s.replace('#include <stdio.h>','#include <stdio.h>\n#include <fstream>\n#include <string>\nstatic std::string capdir(){const char*p=getenv("STUDY_CAPTURE_DIR");return p?p:"/tmp";}\nstatic void meta(const char*name,size_t a,size_t b){std::ofstream f(capdir()+"/metadata.txt",std::ios::app);f<<name<<" "<<a<<" "<<b<<"\\n";}\n')
s=s.replace('  trans->push_end();','  trans->push_end();\n  meta("add_input_history",wireSer,trans->size());')
needle='  cudaMemcpy(d_eTableMgr, h_eTableMgr, sizeof(char) * TRUTH_SIZE * eSize,\n             cudaMemcpyHostToDevice);';assert needle in s
s=s.replace(needle,needle+'\n  {std::vector<char>x(TRUTH_SIZE*eSize);cudaMemcpy(x.data(),d_eTableMgr,x.size(),cudaMemcpyDeviceToHost);std::ofstream f(capdir()+"/actual_truth_bytes.bin",std::ios::binary);f.write(x.data(),x.size());}')
needle='  cudaMemcpy(h_oHisSize, d_oHisSize, sizeof(size_t) * parallelSize,\n             cudaMemcpyDeviceToHost);';assert needle in s
s=s.replace(needle,needle+'\n  {size_t total=0;for(size_t i=0;i<parallelSize;i++)total+=h_oHisSize[i];meta("GPU_output_histories",parallelSize,total);}')
p.write_text(s)
p=dst/'Simulation/src/parser/SAIF.cpp';s=p.read_text().replace('#include "SAIF.h"','#include "SAIF.h"\n#include <fstream>\n#include <string>\n');s=s.replace('    if (His->isDeleted()) return ;','    if (His->isDeleted()) return ;\n    {const char*p=getenv("STUDY_CAPTURE_DIR");std::ofstream f(std::string(p?p:"/tmp")+"/final_wave_counts.txt",std::ios::app);f<<w->getName()<<" "<<His->size()<<"\\n";}');p.write_text(s)
