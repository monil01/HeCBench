#!/usr/bin/env python3
"""Create supplemental untimed orientation fixtures; never modify primary assets."""
import pathlib,json,importlib.util,array,hashlib
P=pathlib.Path(__file__).resolve().parent.parent;s=importlib.util.spec_from_file_location('sw',P/'native_diagnostics/audit_cli.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
cases=[(b,a)for a,b in json.loads((P/'fixture.json').read_text())['cases']];N=len(cases);sa=list(''.join(a for a,b in cases).encode());sb=list(''.join(b for a,b in cases).encode());oa=[];ob=[];a=b=0;outputs=[[]for _ in range(5)]
for x,y in cases:
 a+=len(x);b+=len(y);oa.append(a);ob.append(b);score,(ae,be),_=o.sw(x,y);back,(ac,bc),_=o.sw(x[:ae][::-1],y[:be][::-1],True);assert score==back
 for v,z in zip(outputs,[ae-ac,ae,be-bc,be,score]):v.append(z)
wrap=lambda v,g:[g]*16+v+[g]*16
pre=wrap([0]*len(sa),211)+wrap([0]*len(sb),211)+wrap([0]*N,24681357)+wrap([0]*N,24681357)+wrap(o.mat,-12345)+wrap(o.enc,-12345)
post=wrap(sa,211)+wrap(sb,211)+wrap(oa,24681357)+wrap(ob,24681357)+wrap(o.mat,-12345)+wrap(o.enc,-12345)
actual=[N,128,160,-6,-1,128,min(max(outputs[1]),max(outputs[3]))]+pre+post+outputs[1]+outputs[3]+sum([wrap([-12345]*N,-12345)for _ in range(5)],[])+sum([wrap(v,-12345)for v in outputs],[])
D=P/'native_diagnostics';(D/'orientation_swapped_expected_i32.bin').write_bytes(array.array('i',actual).tobytes());mapping={'NA':'NB','NB':'NA','SEQ_A':'SEQ_B','SEQ_B':'SEQ_A','OFF_A':'OFF_B','OFF_B':'OFF_A'}
lines=[]
for line in (P/'study_params.jl').read_text().splitlines():
 key=line.split('=',1)[0].replace('const ','');lines.append(line.replace('const '+key+'=','const '+mapping.get(key,key)+'=',1))
(D/'orientation_swapped_params.jl').write_text('\n'.join(lines)+'\n')
src=(P/'study_native.cu').read_text().split('int main(int argc,char**argv)')[0];src=src.replace('#include "../bsw-cuda/kernel.cu"','#include "../../bsw-cuda/kernel.cu"').replace('#include "study_params.h"','#include "../study_params.h"\nconstexpr int SWAP_NA=NB,SWAP_NB=NA;\nconst unsigned char *SWAP_A=SEQ_B,*SWAP_B=SEQ_A;\nconst unsigned *SWAP_OA=OFF_B,*SWAP_OB=OFF_A;\n#define NA SWAP_NA\n#define NB SWAP_NB\n#define SEQ_A SWAP_A\n#define SEQ_B SWAP_B\n#define OFF_A SWAP_OA\n#define OFF_B SWAP_OB')
src+='''int main(int argc,char**argv){Work w;std::vector<int> before,initial;w.inputs(before);w.capture_outputs(initial);w.pipeline();std::vector<int> actual={N,128,160,-6,-1,FORWARD,w.reverse};actual.insert(actual.end(),before.begin(),before.end());w.inputs(actual);for(auto v:w.endA)actual.push_back(v);for(auto v:w.endB)actual.push_back(v);actual.insert(actual.end(),initial.begin(),initial.end());w.capture_outputs(actual);std::ifstream f(argv[2],std::ios::binary);std::vector<int> expected(COUNT);f.read((char*)expected.data(),COUNT*4);bool ok=f.gcount()==COUNT*4&&actual==expected;std::ofstream out(argv[1],std::ios::binary);out.write((char*)actual.data(),actual.size()*4);out.close();printf("DIAGNOSTIC swapped_orientation actual_values=%zu correct=%s timing_excluded=true\\n",actual.size(),ok?"true":"false");return ok?0:1;}
''';(D/'orientation_swapped.cu').write_text(src)
(D/'orientation_swapped_fixture.json').write_text(json.dumps(dict(diagnostic_only=True,timing_excluded=True,cases=cases,checked_values=len(actual),expected_sha256=hashlib.sha256(array.array('i',actual).tobytes()).hexdigest(),source_primary=(P/'study_native.cu').name,source_primary_sha256=hashlib.sha256((P/'study_native.cu').read_bytes()).hexdigest(),reason='Exercise actual AshorterthanB devicebranch with unchanged original/translatedkernels; no primarysource/config/artifact modifications.'),indent=2)+'\n')
