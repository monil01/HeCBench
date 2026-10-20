from pathlib import Path
import shutil,json,hashlib
R=Path(__file__).resolve().parents[3];D=R/'.porting-state/diagnostics/logic-resim';out=D/'wave_source';shutil.copytree(D/'source',out,dirs_exist_ok=True)
p=out/'Simulation/src/parser/SAIF.cpp';s=p.read_text().replace('#include "SAIF.h"','#include "SAIF.h"\n#include <fstream>\n#include <cstdint>\n#include <cstdlib>\n');needle='    if (His->isDeleted()) return ;';assert needle in s
s=s.replace(needle,needle+'''\n    { static std::ofstream capture(std::string(getenv("STUDY_CAPTURE_DIR"))+"/original_all_net_waves.bin",std::ios::binary);
      std::string label=w->getName();uint64_t namesize=label.size(),events=His->size();
      capture.write((char*)&namesize,8);capture.write(label.data(),namesize);capture.write((char*)&events,8);
      std::vector<uint64_t> fields;fields.reserve(events*2);
      for(size_t k=0;k<events;k++){fields.push_back((*His)[k].t);fields.push_back((unsigned char)(*His)[k].v);}
      capture.write((char*)fields.data(),fields.size()*8);
    }
''');p.write_text(s)
# Capture CPU-parsed actual initial net histories before graph evaluation, no GPU output used as fixture.
p=out/'Simulation/src/main/main.cpp';s=p.read_text();s='#include <fstream>\n#include <cstdint>\n#include <cstdlib>\n'+s;needle='   vParser.parseAll(vcdFile.getPtr());';assert needle in s
s=s.replace(needle,needle+'''\n   {std::ofstream f(std::string(getenv("STUDY_CAPTURE_DIR"))+"/original_initial_net_waves.bin",std::ios::binary);
    uint64_t total=iParser.getTransC();f.write((char*)&total,8);
    for(uint64_t wire=4;wire<total+4;wire++){tHistory*h=(*iParser.getWireMgr())[wire];uint64_t n=h->size();f.write((char*)&wire,8);f.write((char*)&n,8);std::vector<uint64_t>a;for(size_t k=0;k<n;k++){a.push_back((*h)[k].t);a.push_back((unsigned char)(*h)[k].v);}f.write((char*)a.data(),a.size()*8);}}
''');p.write_text(s)
source=R/'src/logic-resim-cuda/Simulation/src/sim/Simulator.cu';assert (out/'Simulation/src/sim/Simulator.cu').read_bytes()==source.read_bytes()
print('UnmodifiedentireSimulator.cu, CPUinitial/postnetobserveronly')
