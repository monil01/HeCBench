using CUDA, KernelAbstractions
# Source-preserving 32-lane clique predicates. Explicit cooperative barriers
# replace source warp-lockstep communication, without changing integer decisions.
@kernel function delete_kernel!(D,O)
    lane=Int32(@index(Local,Linear)-1); task=Int32(@index(Group,Linear)-1)
    n=Int32(D[1]); nc=Int32(D[2]); dims=Int32(2); cl=dims+nc
    nt=Int32(D[cl+nc*n+1]); aa=cl+nc*n+1; bb=aa+nt
    a=D[aa+task+1]; b=D[bb+task+1]
    s=@localmem Int16 (4,)
    if lane==0;s[1]=0;s[4]=-1;end
    @synchronize
    for c in Int32(0):nc-1
        if lane==0;s[2]=0;s[3]=0;end
        @synchronize
        for j in lane:Int32(32):Int32(D[dims+c+1])-1
            v=D[cl+c*n+j+1]
            if v==a;s[2]=1;end
            if v==b;s[3]=1;end
        end
        @synchronize
        if lane==0 && s[2]!=0 && s[3]!=0
            s[1]+=Int16(1);s[4]=Int16(c)
        end
        @synchronize
        if s[1]>1;break;end
    end
    if lane==0;O[task+1]=s[1]==1 ? s[4] : Int16(-1);end
end
@kernel function add_kernel!(D,A,O,shared_words)
    lane=Int32(@index(Local,Linear)-1); task=Int32(@index(Group,Linear)-1)
    n=Int32(D[1]); nc=Int32(D[2]); dims=Int32(2); cl=dims+nc
    ns=Int32(A[n+1]); sd=n+1; sep=sd+ns
    ne=Int32(A[sep+n*ns+1]); ta=sep+n*ns+1; tb=ta+ne; edge=tb+ne
    nt=Int32(A[edge+n*n+1]); aa=edge+n*n+1; bb=aa+nt
    s=@localmem Int16 (12,)
    w=@localmem Int16 (180,)
    rr=Int32(0); tt=ne; ss=2ne; cs=2ne+n
    if lane==0
        s[11]=0;s[6]=-1;s[7]=-1;s[8]=0;s[9]=A[aa+task+1];s[10]=A[bb+task+1];O[task+1]=0
        if A[Int32(s[9])+1]!=A[Int32(s[10])+1];s[11]=1;O[task+1]=1;end
    end
    @synchronize
    if s[11]==0
        if lane==0
            s[1]=0
            for j in Int32(0):n-1
                if A[edge+Int32(s[9])*n+j+1]!=0 && A[edge+Int32(s[10])*n+j+1]!=0
                    w[ss+Int32(s[1])+1]=Int16(j);s[1]+=Int16(1)
                end
            end
            if s[1]==0;s[11]=1;end
        end
        @synchronize
        if s[11]==0
            for c in Int32(0):nc-1
                if lane==0;s[4]=0;s[5]=0;end
                w[cs+lane+1]=0;@synchronize
                for j in lane:Int32(32):Int32(D[dims+c+1])-1
                    v=D[cl+c*n+j+1]
                    if v==s[9];s[4]=1;end
                    if v==s[10];s[5]=1;end
                    for k in Int32(0):Int32(s[1])-1
                        if v==w[ss+k+1];w[cs+lane+1]+=Int16(1);end
                    end
                end
                @synchronize
                if lane==0
                    count=Int32(0)
                    for j in Int32(0):Int32(31);count+=Int32(w[cs+j+1]);end
                    if s[4]!=0 && count==Int32(s[1]);s[6]=Int16(c);end
                    if s[5]!=0 && count==Int32(s[1]);s[7]=Int16(c);end
                end
                @synchronize
            end
            if lane==0
                w[rr+1]=-1;s[2]=-1
                for i in Int32(0):ne-1
                    if A[ta+i+1]==s[6];w[rr+1]=Int16(i);s[2]=0;break;end
                end
                for i in Int32(w[rr+1]):-Int32(1):Int32(0)
                    if A[ta+i+1]==A[tb+Int32(w[rr+Int32(s[2])+1])+1]
                        s[2]+=Int16(1);w[rr+Int32(s[2])+1]=Int16(i)
                    end
                end
            elseif lane==1
                w[tt+1]=-1;s[3]=-1
                for i in Int32(0):ne-1
                    if A[ta+i+1]==s[7];w[tt+1]=Int16(i);s[3]=0;break;end
                end
                for i in Int32(w[tt+1]):-Int32(1):Int32(0)
                    if A[ta+i+1]==A[tb+Int32(w[tt+Int32(s[3])+1])+1]
                        s[3]+=Int16(1);w[tt+Int32(s[3])+1]=Int16(i)
                    end
                end
            end
            @synchronize
            if lane==0
                t=min(Int32(s[2]),Int32(s[3]))
                for i in Int32(0):t
                    if A[tb+Int32(w[rr+Int32(s[2])-i+1])+1]==A[tb+Int32(w[tt+Int32(s[3])-i+1])+1];s[8]=Int16(i);else;break;end
                end
                if t!=-1 && A[ta+Int32(w[rr+Int32(s[2])-Int32(s[8])+1])+1]==A[ta+Int32(w[tt+Int32(s[3])-Int32(s[8])+1])+1];s[8]+=Int16(1);end
            end
            @synchronize
            found=false
            for i in lane:Int32(32):(Int32(s[2])-Int32(s[8]))+(Int32(s[3])-Int32(s[8]))+1
                path=i<=Int32(s[2])-Int32(s[8]) ? Int32(w[rr+i+1]) : Int32(w[tt+i-(Int32(s[2])-Int32(s[8]))-1+1])
                if A[sd+path+1]==s[1]
                    count=Int32(0)
                    for j in Int32(0):Int32(s[1])-1, k in Int32(0):Int32(s[1])-1
                        if A[sep+path*n+j+1]==w[ss+k+1];count+=1;end
                    end
                    found|=count==Int32(s[1])
                end
            end
            w[cs+lane+1]=found ? Int16(1) : Int16(0);@synchronize
            if lane==0
                any_found=false
                for j in Int32(0):Int32(31);any_found|=w[cs+j+1]!=0;end
                O[task+1]=any_found ? Int16(1) : Int16(0)
            end
            @synchronize
        end
    end
end

function launch_delete(D,O,tasks,shmem,stream)
    CUDA.stream!(stream)
    delete_kernel!(get_backend(D),32)(D,O;ndrange=tasks*32)
end
function launch_add(D,A,O,tasks,shmem,stream)
    shmem<=360 || error("Named50vertexKAworkspace exceeded")
    CUDA.stream!(stream)
    add_kernel!(get_backend(D),32)(D,A,O,Int32(shmem÷2);ndrange=tasks*32)
end
include(joinpath(@__DIR__,"..","study_driver.jl"))
main()
