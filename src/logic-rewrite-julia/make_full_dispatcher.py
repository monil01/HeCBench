#!/usr/bin/env python3
"""Build an explicit ABI dispatcher from reviewed source signatures.
No unsupported source kernel has a native fallback.
"""
import pathlib,json,re
B=pathlib.Path(__file__).resolve().parent;J=B/'logic-rewrite-julia-cuda.jl';available=set()
for p in J.glob('*.jl'):available.update(re.findall(r'function lr_([A-Za-z_0-9]+)\(',p.read_text()))
rows={r['name']:r for r in json.loads((B/'source_audit/inventory.json').read_text())['definitions']if r['kind']=='__global__'and r['name']in available and r['parameters']}
def row(name,types):rows[name]=dict(name=name,parameters=[dict(type=t)for t in types])
row('recordMFFC',['int *']*8+['int']*3)
row('getCutTruthTableConsecutive',['int *']*6+['unsigned int *','int *','unsigned int *']+['int']*3+['int *'])
row('gatherTableToConsecutive',['int *']*5+['int'])
row('printStatsKernel',['int *']*3)
maptype={'int':'Int32','unsigned int':'UInt32','uint64':'UInt64','uint32':'UInt32','rw::Cut':'LRCut','Cut':'LRCut','rw::Library':'LRLibrary','Library':'LRLibrary','rw::TableNode':'LRTableNode','TableNode':'LRTableNode','bool':'Bool','unsigned long long':'UInt64'}
cases=[]
for index,(name,r)in enumerate(sorted(rows.items())):
 args=[]
 for k,param in enumerate(r['parameters'],1):
  t=' '.join(param['type'].replace('const','').split());pointer='*'in t;t=t.replace('*','').strip();typ=maptype[t]
  args.append('p('+str(k)+','+typ+')'if pointer else'lr_cast('+typ+',v['+str(k)+'])')
 cases.append(('        if'if index==0 else'        elseif')+' base=="'+name+'"\n            @assert n=='+str(len(args))+';f=lr_'+name+';args=('+','.join(args)+',)')
for typ,label in [('UInt64','hash64_32'),('UInt32','hash32_32')]:
 for name,types in [('getEntryIndicator',['Int32',typ,'i']),('gatherKeyValues',['Int32',typ,'UInt32',typ,'UInt32','i']),('insert_batch_no_update_kernel',[typ,'UInt32',typ,'UInt32','i','i']),('insert_batch_no_update_masked_kernel',[typ,'UInt32',typ,'UInt32','Int32','i','i']),('retrieve_batch_kernel',[typ,'UInt32',typ,'UInt32','i','i']),('retrieve_batch_masked_kernel',[typ,'UInt32',typ,'UInt32','Int32','i','i'])]:
  args=['i('+str(k)+')'if t=='i'else'p('+str(k)+','+t+')'for k,t in enumerate(types,1)]
  cases.append('        elseif name=="'+label+'.'+name+'"\n            @assert n=='+str(len(args))+';f=lr_s_'+name+';args=('+','.join(args)+',)')
text='''# Qualified hybrid: authored Julia explicit kernels, original C++ CPU controller
# and native Thrust host library operations/predicates. No native kernel fallback.
using CUDA,Libdl,SHA
'''+''.join('include("'+p+'")\n'for p in ['c_memory.jl','c_library.jl','generated_cut_helpers.jl','strash_helpers.jl','generated_strash.jl','balance_helpers.jl','generated_balance.jl','factor_helpers.jl','generated_factor.jl','mffc_helpers.jl','generated_mffc.jl','manager_kernels.jl'])+'''
CUDA.device!(0);CUDA.context()
const LR_LIBRARY_PATH=abspath(ARGS[1]);const LR_LIB=Libdl.dlopen(LR_LIBRARY_PATH)
const LR_RUN=Libdl.dlsym(LR_LIB,:lr_full_study);const LR_SYMBOL=Libdl.dlsym(LR_LIB,:lr_symbol)
const LR_GLOBALS=(P=lr_ptr(Int32,UInt64(UInt(ccall(LR_SYMBOL,Ptr{Cvoid},(Cint,),0)))),N=lr_ptr(Int32,UInt64(UInt(ccall(LR_SYMBOL,Ptr{Cvoid},(Cint,),1)))))
@assert UInt(getfield(LR_GLOBALS.P,:base))!=0&&UInt(getfield(LR_GLOBALS.N,:base))!=0
const LR_COUNTS=Dict{String,Int}()
Base.@noinline function lr_launch_device(f,args,threads,grid)
    CUDA.@cuda threads=threads blocks=grid stream=CUDA.legacy_stream() maxregs=128 f(LR_GLOBALS,args...)
    return nothing
end
function lr_callback(d::Ptr{Cvoid})::Cint
 name="packet_unparsed"
 try
    name=unsafe_string(unsafe_load(Ptr{Cstring}(d)));base=split(split(name,'.')[end],"::")[end];words=Ptr{UInt64}(d)
    grid=ntuple(i->Int(unsafe_load(words,1+i)),3);threads=ntuple(i->Int(unsafe_load(words,4+i)),3);n=Int(unsafe_load(words,8));@assert n<=32
    v=ntuple(i->unsafe_load(words,40+i),n);i(k)=lr_cast(Int32,v[k]);p(k,T)=lr_ptr(T,v[k])
'''+ '\n'.join(cases)+'''
        else
            error("No authored Julia implementation for intercepted explicit source kernel: ",name)
        end
    Base.invokelatest(lr_launch_device,f,args,threads,grid)
    LR_COUNTS[name]=get(LR_COUNTS,name,0)+1
    return Cint(0)
 catch err
    println(stderr,"JULIA_SOURCE_KERNEL_REJECTED name=",name," completed_dispatches=",LR_COUNTS)
    showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr);return Cint(1)
 end
end
precompile(lr_callback,(Ptr{Cvoid},));const LR_CALLBACK=@cfunction(lr_callback,Cint,(Ptr{Cvoid},))
println("STUDY_RUNTIME_BINDING wrapper_sha256=",bytes2hex(sha256(read(LR_LIBRARY_PATH)))," explicit_kernel_native_fallback=false; original_native_host_Thrust_library_interoperation=true")
status=ccall(LR_RUN,Cint,(Cstring,Ptr{Cvoid}),abspath(ARGS[2]),LR_CALLBACK)
CUDA.device_synchronize();println("JULIA_EXPLICIT_KERNEL_DISPATCH_COUNTS ",LR_COUNTS)
status==0||error("Full source pipeline failed its declared checks/status ",status)
'''
(J/'main.jl').write_text(text);(B/'full_dispatch_signatures.json').write_text(json.dumps(rows,indent=2)+'\n');print('Explicit source dispatcher cases',len(rows),'plus12typedhashlibraryentrycases')
