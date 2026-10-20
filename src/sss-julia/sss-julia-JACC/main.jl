using CUDA, JACC
# Source-preserving 32-lane clique predicates. Explicit cooperative barriers
# replace source warp-lockstep communication, without changing integer decisions.
@inline function delete_kernel!(i,D,O,template)
    lane=Int32(rem(i-1,32)); task=Int32(div(i-1,32))
    n=Int32(D[1]); nc=Int32(D[2]); dims=Int32(2); cl=dims+nc
    nt=Int32(D[cl+nc*n+1]); aa=cl+nc*n+1; bb=aa+nt
    a=D[aa+task+1]; b=D[bb+task+1]
    s=JACC.shared(template)
    if lane==0;s[1]=0;s[4]=-1;end
    JACC.sync_workgroup()
    for c in Int32(0):nc-1
        if lane==0;s[2]=0;s[3]=0;end
        JACC.sync_workgroup()
        for j in lane:Int32(32):Int32(D[dims+c+1])-1
            v=D[cl+c*n+j+1]
            if v==a;s[2]=1;end
            if v==b;s[3]=1;end
        end
        JACC.sync_workgroup()
        if lane==0 && s[2]!=0 && s[3]!=0
            s[1]+=Int16(1);s[4]=Int16(c)
        end
        JACC.sync_workgroup()
        if s[1]>1;break;end
    end
    if lane==0;O[task+1]=s[1]==1 ? s[4] : Int16(-1);end
    return
end
@inline function add_kernel!(i,D,A,O,template)
    lane=Int32(rem(i-1,32)); task=Int32(div(i-1,32))
    n=Int32(D[1]); nc=Int32(D[2]); dims=Int32(2); cl=dims+nc
    ns=Int32(A[n+1]); sd=n+1; sep=sd+ns
    ne=Int32(A[sep+n*ns+1]); ta=sep+n*ns+1; tb=ta+ne; edge=tb+ne
    nt=Int32(A[edge+n*n+1]); aa=edge+n*n+1; bb=aa+nt
    w=JACC.shared(template)
    rr=Int32(0); tt=ne; ss=2ne; cs=2ne+n
    if lane==0
        w[191]=0;w[186]=-1;w[187]=-1;w[188]=0;w[189]=A[aa+task+1];w[190]=A[bb+task+1];O[task+1]=0
        if A[Int32(w[189])+1]!=A[Int32(w[190])+1];w[191]=1;O[task+1]=1;end
    end
    JACC.sync_workgroup()
    if w[191]==0
        if lane==0
            w[181]=0
            for j in Int32(0):n-1
                if A[edge+Int32(w[189])*n+j+1]!=0 && A[edge+Int32(w[190])*n+j+1]!=0
                    w[ss+Int32(w[181])+1]=Int16(j);w[181]+=Int16(1)
                end
            end
            if w[181]==0;w[191]=1;end
        end
        JACC.sync_workgroup()
        if w[191]==0
            for c in Int32(0):nc-1
                if lane==0;w[184]=0;w[185]=0;end
                w[cs+lane+1]=0;JACC.sync_workgroup()
                for j in lane:Int32(32):Int32(D[dims+c+1])-1
                    v=D[cl+c*n+j+1]
                    if v==w[189];w[184]=1;end
                    if v==w[190];w[185]=1;end
                    for k in Int32(0):Int32(w[181])-1
                        if v==w[ss+k+1];w[cs+lane+1]+=Int16(1);end
                    end
                end
                JACC.sync_workgroup()
                if lane==0
                    count=Int32(0)
                    for j in Int32(0):Int32(31);count+=Int32(w[cs+j+1]);end
                    if w[184]!=0 && count==Int32(w[181]);w[186]=Int16(c);end
                    if w[185]!=0 && count==Int32(w[181]);w[187]=Int16(c);end
                end
                JACC.sync_workgroup()
            end
            if lane==0
                w[rr+1]=-1;w[182]=-1
                for i in Int32(0):ne-1
                    if A[ta+i+1]==w[186];w[rr+1]=Int16(i);w[182]=0;break;end
                end
                for i in Int32(w[rr+1]):-Int32(1):Int32(0)
                    if A[ta+i+1]==A[tb+Int32(w[rr+Int32(w[182])+1])+1]
                        w[182]+=Int16(1);w[rr+Int32(w[182])+1]=Int16(i)
                    end
                end
            elseif lane==1
                w[tt+1]=-1;w[183]=-1
                for i in Int32(0):ne-1
                    if A[ta+i+1]==w[187];w[tt+1]=Int16(i);w[183]=0;break;end
                end
                for i in Int32(w[tt+1]):-Int32(1):Int32(0)
                    if A[ta+i+1]==A[tb+Int32(w[tt+Int32(w[183])+1])+1]
                        w[183]+=Int16(1);w[tt+Int32(w[183])+1]=Int16(i)
                    end
                end
            end
            JACC.sync_workgroup()
            if lane==0
                t=min(Int32(w[182]),Int32(w[183]))
                for i in Int32(0):t
                    if A[tb+Int32(w[rr+Int32(w[182])-i+1])+1]==A[tb+Int32(w[tt+Int32(w[183])-i+1])+1];w[188]=Int16(i);else;break;end
                end
                if t!=-1 && A[ta+Int32(w[rr+Int32(w[182])-Int32(w[188])+1])+1]==A[ta+Int32(w[tt+Int32(w[183])-Int32(w[188])+1])+1];w[188]+=Int16(1);end
            end
            JACC.sync_workgroup()
            found=false
            for i in lane:Int32(32):(Int32(w[182])-Int32(w[188]))+(Int32(w[183])-Int32(w[188]))+1
                path=i<=Int32(w[182])-Int32(w[188]) ? Int32(w[rr+i+1]) : Int32(w[tt+i-(Int32(w[182])-Int32(w[188]))-1+1])
                if A[sd+path+1]==w[181]
                    count=Int32(0)
                    for j in Int32(0):Int32(w[181])-1, k in Int32(0):Int32(w[181])-1
                        if A[sep+path*n+j+1]==w[ss+k+1];count+=1;end
                    end
                    found|=count==Int32(w[181])
                end
            end
            w[cs+lane+1]=found ? Int16(1) : Int16(0);JACC.sync_workgroup()
            if lane==0
                any_found=false
                for j in Int32(0):Int32(31);any_found|=w[cs+j+1]!=0;end
                O[task+1]=any_found ? Int16(1) : Int16(0)
            end
            JACC.sync_workgroup()
        end
    end
    return
end

# Extra framework preload templates are read-only implementation inputs, not
# numerical problem inputs. Their actual complete bytes and physical guards
# are read back before and after all full source searches.
const DELETE_HOST=vcat(fill(Int16(0x5a5a),17),zeros(Int16,4),fill(Int16(0x5a5a),17))
const ADD_HOST=vcat(fill(Int16(0x5a5a),17),zeros(Int16,192),fill(Int16(0x5a5a),17))
const DELETE_PARENT=CuArray(DELETE_HOST)
const ADD_PARENT=CuArray(ADD_HOST)
const DELETE_TEMPLATE=unsafe_wrap(CuArray,pointer(DELETE_PARENT,18),(4,);own=false)
const ADD_TEMPLATE=unsafe_wrap(CuArray,pointer(ADD_PARENT,18),(192,);own=false)
const SPECS=Dict{Tuple{Int,Int,UInt},Any}()
function launch_delete(D,O,tasks,shmem,stream)
    spec=get!(SPECS,(0,tasks,UInt(stream.handle))) do
        JACC.launch_spec(threads=32,blocks=tasks,shmem_size=8,sync=false,stream=stream)
    end
    JACC.parallel_for(spec,tasks*32,delete_kernel!,D,O,DELETE_TEMPLATE)
end
function launch_add(D,A,O,tasks,shmem,stream)
    shmem<=360 || error("Named50vertexJACCworkspace exceeded")
    spec=get!(SPECS,(1,tasks,UInt(stream.handle))) do
        JACC.launch_spec(threads=32,blocks=tasks,shmem_size=384,sync=false,stream=stream)
    end
    JACC.parallel_for(spec,tasks*32,add_kernel!,D,A,O,ADD_TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_driver.jl"))
function template_readback(path)
    delete_actual=Array(DELETE_PARENT);add_actual=Array(ADD_PARENT)
    delete_actual==DELETE_HOST || error("JACCdelete-template actual bytes/redzones changed")
    add_actual==ADD_HOST || error("JACCadd-template actual bytes/redzones changed")
    open(path,"w") do io
        print(io,"{\"delete_actual_i16\":[",join(delete_actual,","),"],\"add_actual_i16\":[",join(add_actual,","),"],\"physical_guard_words\":68,\"all_read_only_template_bytes_checked\":true}")
    end
end
template_readback(ARGS[3]*"_template_inputs_before.json")
main()
template_readback(joinpath(ARGS[3],"implementation_template_inputs_after.json"))
