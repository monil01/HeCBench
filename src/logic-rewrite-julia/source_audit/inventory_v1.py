#!/usr/bin/env python3
"""Read-only Clang AST source inventory, not a C++ or GPU compilation.
Requires clang==18.1.8 Python bindings and an existing libclang18.
"""
import argparse,hashlib,json,pathlib,re,sys
p=argparse.ArgumentParser();p.add_argument('--bindings',required=True);p.add_argument('--libclang',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();sys.path.insert(0,a.bindings)
from clang import cindex
cindex.Config.set_library_file(a.libclang)
root=pathlib.Path(__file__).resolve().parents[1]/'reference_history'
pre='''#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __managed__
#define __shared__
#include <cuda_runtime.h>
struct __study_dim {unsigned x,y,z;};
extern __study_dim blockIdx,blockDim,threadIdx,gridDim;
void __syncthreads();
template<class T,class U>T atomicAdd(T*,U);
template<class T,class U>T atomicExch(T*,U);
template<class T,class U>T atomicSub(T*,U);
template<class T,class U,class V>T atomicCAS(T*,U,V);
'''
rows=[];diagnostics=[];structs={};sources={}
for f in sorted(root.glob('*.cu')):
    s=f.read_text();sources[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    # Ordinary C++ AST only: replace launch syntax, preserve argument expressions.
    t=re.sub(r'<<<.*?>>>','',s,flags=re.S);name=str(f.with_suffix('.ast.cpp'))
    tu=cindex.Index.create().parse(name,args=['-x','c++','-std=c++17','-I'+str(root),'-I/usr/include/c++/12','-I/usr/include/x86_64-linux-gnu/c++/12','-isystem','/usr/lib/gcc/x86_64-linux-gnu/12/include','-ferror-limit=0'],unsaved_files=[(name,pre+t)])
    diagnostics.extend(dict(source=f.name,severity=d.severity,text=str(d)) for d in tu.diagnostics)
    for c in tu.cursor.walk_preorder():
        if c.kind==cindex.CursorKind.STRUCT_DECL and c.is_definition() and c.location.file and str(root) in c.location.file.name:
            if c.spelling in ('Cut','TableNode','Library'):
                structs[c.spelling]=dict(size=c.type.get_size(),alignment=c.type.get_align(),fields=[dict(name=x.spelling,type=x.type.spelling,bit_offset=x.get_field_offsetof(),bitfield_width=x.get_bitfield_width() if x.is_bitfield() else None) for x in c.get_children() if x.kind==cindex.CursorKind.FIELD_DECL])
        if c.kind not in (cindex.CursorKind.FUNCTION_DECL,cindex.CursorKind.FUNCTION_TEMPLATE) or not c.is_definition() or not c.location.file or c.location.file.name!=name:continue
        match=re.search(r'\b(__global__|__device__)\s+(?:(?:__forceinline__|inline|__host__)\s+)*[^;{}]*?\b'+re.escape(c.spelling)+r'\s*\(',s)
        if not match:continue
        calls=sorted(set(x.referenced.spelling if x.referenced else x.spelling for x in c.walk_preorder() if x.kind==cindex.CursorKind.CALL_EXPR))
        rows.append(dict(source=f.name,name=c.spelling,kind=match.group(1),result=c.result_type.spelling,parameters=[dict(name=x.spelling,type=x.type.spelling) for x in c.get_arguments()],calls=calls,contains_recursive_self_call=c.spelling in calls))
result=dict(purpose='source readiness inventory only; CUDA qualifiers/launch syntax removed for host AST. This is not a CUDA or Julia compiler support test.',original_source_sha256=sources,definitions=rows,struct_layouts=structs,diagnostics=diagnostics,global_definitions=sum(r['kind']=='__global__' for r in rows),device_definitions=sum(r['kind']=='__device__' for r in rows))
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ('global_definitions','device_definitions','struct_layouts')},indent=2));print('AST errors',sum(d['severity']>=3 for d in diagnostics))
