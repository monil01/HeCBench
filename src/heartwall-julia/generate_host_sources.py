from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'study_native.cu').read_text();layout=json.loads((p/'buffer_layout.json').read_text())['buffers']
check=s[s.index('auto check='):s.index('// Full11frame')]
loads=s[s.index('auto expected='):s.index('std::vector<std::unique_ptr<Field>>f;')]
# Lazy initialization once per process; checker is verification only, never in timed GPU ROI.
head='''#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstring>
#include "study_params.h"
#include "native_field_offsets.h"
template<class T>std::vector<T>read(const std::string&p,size_t count=0){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)exit(2);if(!count)count=f.tellg()/sizeof(T);f.seekg(0);std::vector<T>x(count);f.read((char*)x.data(),count*sizeof(T));if(size_t(f.gcount())!=count*sizeof(T))exit(3);return x;}
struct Field{int n;};
extern "C" int check_heartwall(const float*ptr,int count,int frame,int last,double*out_error){
'''
loads=loads.replace('auto ','static auto ')
f='std::vector<std::unique_ptr<Field>>f;\n'+''.join('f.emplace_back(new Field{'+str(b['size'])+'});\n' for b in layout)
(p/'study_checker.cpp').write_text(head+loads+f+check+'std::vector<float>x(ptr,ptr+count); double error=0;bool good=check(x,frame,last,error);*out_error=error;return good?1:0;\n}\n')
fields=json.loads((p/'params_schema.json').read_text())['fields']
constants='include(joinpath(@__DIR__,"study_params.jl"))\nconst FIELD_SPECS=(\n'+''.join('    (:'+b['name']+','+('Int32' if b['type']=='int32' else 'Float32')+','+str(b['size'])+','+str(b['guard'])+'),\n' for b in layout)+')\n'
constants+='const COMMON_NAMES=('+','.join(':'+f['name'] for f in fields)+',)\n'
(p/'host_specs.jl').write_text(constants)
