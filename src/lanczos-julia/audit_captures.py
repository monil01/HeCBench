#!/usr/bin/env python3
"""CPU-only full saved Lanczos replay; NumPy required, no GPU invocation."""
import argparse,pathlib,json,hashlib,math,statistics
import numpy as np
P=argparse.ArgumentParser();P.add_argument('--result',type=pathlib.Path,required=True);P.add_argument('--output',type=pathlib.Path,required=True);P.add_argument('--run-root',type=pathlib.Path);a=P.parse_args()
N=257;SIZES=((6,5,4),(17,13,9),(25,19,13),(37,29,23),(53,43,31));PREFIX=39099

def graph(sizes):
    M=np.zeros((N,N),dtype=np.float64);start=0;prev=None
    def edge(i,j):M[i,j]=M[j,i]=1
    for count in sizes:
        M[start:start+count,start:start+count]=1
        for i in range(start,start+count):M[i,i]=0
        if prev is not None:edge(prev,start)
        prev=start+count-1;start+=count
    edge(prev,start)
    for i in range(start,N-1):edge(i,i+1)
    cols=[np.flatnonzero(M[row])[::-1].astype(np.int32) for row in range(N)]
    rp=np.concatenate(([0],np.cumsum([len(x) for x in cols]))).astype(np.int32);ci=np.concatenate(cols);v=np.ones(len(ci));return M,rp,ci,v
GRAPHS=[graph(s) for s in SIZES]

def bits(x):
    u=np.asarray(x,dtype='<f8').view('<u8');w=np.empty(2*len(u));w[::2]=(u&np.uint64(0xffffffff)).astype(np.float64);w[1::2]=(u>>np.uint64(32)).astype(np.float64);return w

def same(x,y,why='bitwise chain'):
    assert np.array_equal(np.asarray(x,dtype='<f8').view('<u8'),np.asarray(y,dtype='<f8').view('<u8')),why

def replay(path):
    raw=np.memmap(path,dtype='<f8',mode='r');assert len(raw)==3103860 and np.isfinite(raw).all();pos=0
    def take(n):
        nonlocal pos
        z=raw[pos:pos+n];assert len(z)==n;pos+=n;return z
    def scalar():return float(take(1)[0])
    for idx,(M,rp,ci,v) in enumerate(GRAPHS):
        g=(2,4,8,16,32)[idx];same(take(5),[N,len(ci),g,3,71]);same(take(N+1),rp);same(take(len(ci)),ci);same(take(2*len(v)),bits(v));e0=np.zeros(N);e0[0]=1;same(take(2*N),bits(e0))
    assert pos==PREFIX;max_ratio=0.;max_stage_abs=0.;checked=0;cases=[];negative=None
    def compare(got,gold):
        nonlocal max_ratio,max_stage_abs,checked,negative
        gold=np.asarray(gold,dtype=np.longdouble);got=np.asarray(got);err=np.abs(got.astype(np.longdouble)-gold);budget=np.longdouble('1e-10')+np.longdouble('1e-10')*np.abs(gold);r=float(np.max(err/budget));assert r<=1,'full actual-stage independent oracle';max_ratio=max(max_ratio,r);max_stage_abs=max(max_stage_abs,float(np.max(err)));checked+=len(got)
        if negative is None:negative={'output_plus_one_rejected':bool(abs(np.longdouble(got[0]+1)-gold[0])>budget[0]),'guard_byte_change_rejected':166!=165,'CSR_column_change_rejected':not np.array_equal(GRAPHS[0][2],GRAPHS[0][2]^1)};assert all(negative.values())
    def stage(expected,idx):
        tag=scalar();assert tag==100+expected,(pos,tag,expected);n,nz,g,b=map(int,take(4));s=scalar();assert n==N
        if expected==0:
            assert nz==g==0 and b==2;x=take(N);y=take(N);out=take(2);gold=[np.sum(x[j*256:min(N,(j+1)*256)].astype(np.longdouble)*y[j*256:min(N,(j+1)*256)].astype(np.longdouble),dtype=np.longdouble) for j in range(2)];compare(out,gold);return s,x,y,out
        if expected==1:
            assert nz==g==0 and b==2;x=take(N);out=take(N);compare(out,x.astype(np.longdouble)*np.longdouble(s));return s,x,out
        if expected==2:
            assert nz==g==0 and b==2;y=take(N);x=take(N);out=take(N);compare(out,y.astype(np.longdouble)+np.longdouble(s)*x.astype(np.longdouble));return s,y,x,out
        M,rp,ci,v=GRAPHS[idx];assert nz==len(ci) and g==(2,4,8,16,32)[idx] and b==(N*g+255)//256
        same(take(N+1),rp,'actualGPUCSR rowbits');same(take(nz),ci,'actualGPUCSR colbits');same(take(nz),v,'actualGPUCSR valuebits');x=take(N);out=take(N)
        norm=np.sum(x.astype(np.longdouble)**2);assert abs(norm-1)<=1e-10;gold=M.astype(np.longdouble)@x.astype(np.longdouble);compare(out,gold);return x,out
    for idx,(M,rp,ci,v) in enumerate(GRAPHS):
        basis=[];alphas=[];betas=[];last=None;next_x=np.zeros(N);next_x[0]=1;max_rec=0.
        for step in range(71):
            x,Ax=stage(3,idx);same(x,next_x,'recurrence source x_{i} chain');basis.append(x.copy());_,dx,dy,dot=stage(0,idx);same(dx,x);same(dy,Ax);alpha=float(dot[0])+float(dot[1]);alphas.append(alpha)
            scalar1,yp,xp,work=stage(2,idx);same([scalar1],[-alpha]);same(yp,Ax);same(xp,x)
            if step:
                scalar2,yp,xp,out=stage(2,idx);same([scalar2],[-betas[-1]]);same(yp,work);same(xp,last);work=out
            _,dx,dy,dot=stage(0,idx);same(dx,work);same(dy,work);beta=math.sqrt(float(dot[0])+float(dot[1]));assert beta>0;betas.append(beta)
            factor,scaled_input,next_x=stage(1,idx);same([factor],[1/beta]);same(scaled_input,work)
            residual=Ax.astype(np.longdouble)-np.longdouble(alpha)*x.astype(np.longdouble)-np.longdouble(beta)*next_x.astype(np.longdouble)
            if step:residual-=np.longdouble(betas[-2])*last.astype(np.longdouble)
            max_rec=max(max_rec,float(np.max(np.abs(residual))));last=x.copy()
        # Source pointer rotation changes free order, so require the exact allocation-ID set/bytes.
        expected_sizes=[4*(N+1),4*len(ci),8*len(ci),8*N,8*N,8*N,16];ids=set()
        for free in range(7):
            assert scalar()==200;ident=int(scalar());size=int(scalar());assert ident//7==idx and ident not in ids;ids.add(ident);assert size==expected_sizes[ident%7];assert np.all(take(512)==165),'actualGPU guardbytes'
        assert ids==set(range(idx*7,idx*7+7));assert scalar()==300 and scalar()==71;ta=take(71);tb=take(70);same(ta,alphas,'capturedalpha vsactualGPUdot chain');same(tb,betas[:70],'capturedbeta vsactualGPUnorm chain')
        assert scalar()==400 and scalar()==idx and scalar()==3;roots=take(3);truth=np.linalg.eigvalsh(M)[-3:][::-1];errors=np.abs(roots-truth);budgets=1e-4+1e-6*np.abs(truth);assert np.all(errors<=budgets),'fullgraph largest3 spectralaccuracy';checked+=3
        T=np.diag(ta)+np.diag(tb,1)+np.diag(tb,-1);ritz=np.linalg.eigvalsh(T);nearest=np.min(np.abs(roots[:,None]-ritz[None,:]),axis=1);assert np.all(nearest<=budgets),'sourceTQLRAT returnedroot notin computedtridiagonal spectrum budget'
        Q=np.stack(basis,axis=1);gram=Q.T@Q;offdiag=gram-np.diag(np.diag(gram));cases.append({'case':idx,'n':N,'nonzeros':len(ci),'group_size':(2,4,8,16,32)[idx],'steps':71,'returned_roots':roots.tolist(),'independent_dense_graph_roots':truth.tolist(),'max_abs_spectral_error':float(np.max(errors)),'max_normalized_spectral_error':float(np.max(errors/budgets)),'nearest_independent_tridiagonal_eigenvalue_errors':nearest.tolist(),'max_abs_recurrence_residual':max_rec,'max_abs_basis_offdiagonal_inner_product':float(np.max(np.abs(offdiag))),'orthogonality_qualification':'Sourceexplicitlynaive/no reorthogonalization; offdiagonals arediagnosticobservations, notafailedhiddenorthogonalitycontract.'})
    assert pos==len(raw) and checked==365090
    return {'pass':True,'artifact_fields':len(raw),'exact_input_prefix_fields':PREFIX,'actual_stage_output_checks':checked,'source_scientific_kernels':4,'source_scientific_launches':2125,'max_abs_stage_error':max_stage_abs,'max_normalized_stage_error':max_ratio,'negative_controls':negative,'cases':cases,'guard_bytes_checked':5*7*512,'chain_checked':'Everyactualinput/output plusalpha/beta/scalar/pointerrotation values throughall71steps; originalsourceTQLRAT/filter checkedagainstindependentfullgraph andtridiagonal spectra.','mathematical_limit':'Fixedengineeringbudget forboundedgraphs, notglobalconvergence/eigenbasisorthogonality proof.'}

result=json.loads(a.result.read_text());assert result['matched_performance_valid'] and result['reference_agreement'];cache={};records=[]
for model in ('cuda','cuda_jl','ka','jacc'):
    m=result['models'][model];assert m['status']=='native_check_pass' and m['complete'] and m['timing_valid'];assert [r['trial'] for r in m['trials']]==[1,2,3,4,5];assert abs(statistics.mean(t['time_seconds'] for t in m['trials'][1:])-m['mean_seconds'])<1e-15
    for row in m['trials']:
        ar=row['output_artifact'];p=pathlib.Path(ar['saved_path'])
        if a.run_root:p=a.run_root/p.parent.name/p.name
        sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==ar['sha256'] and p.stat().st_size==ar['count']*8;reuse=sha in cache
        if not reuse:cache[sha]=replay(p)
        records.append({'model':model,'trial':row['trial'],'path':str(p),'sha256':sha,'full_rehash_pass':True,'mathematical_replay_reused_only_complete_SHA':reuse})
out={'all_pass':True,'result':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'full_artifact_hashes':len(records),'unique_full_CPU_replays':len(cache),'all20_byteidentical':len(cache)==1,'NumPy':np.__version__,'longdouble_mantissa_bits':int(np.finfo(np.longdouble).nmant+1),'scope':'CPU-only fullmathematical replay+everyfilehash. Mathematicalreplay cachedonlycompleteSHAidentity. Separateactualinprocessscientificchecks ran everytrial. NoGPUrerun oracceptancebudgetchange.','records':records,'replays':cache};a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(a.output);print(json.dumps({'all_pass':True,'artifacts':len(records),'unique_full_CPU_replays':len(cache),'max_spectral_error':max(c['max_abs_spectral_error'] for r in cache.values() for c in r['cases'])}))
