#!/usr/bin/env python3
"""CPU-only full saved HPL evidence replay. Requires NumPy; never launches GPU.
Each file is fully rehashed; mathematical replay is reused only for identical SHA.
"""
import argparse,hashlib,json,pathlib,statistics
import numpy as np
P=argparse.ArgumentParser();P.add_argument('--result',type=pathlib.Path,required=True);P.add_argument('--output',type=pathlib.Path,required=True);P.add_argument('--run-root',type=pathlib.Path);args=P.parse_args()
CASES=((1536,512,1544),(257,64,264));PREFIX=4886102

def original_matrix(n,ld):
    # Independent Python arbitrary-integer LCG, not original split-word C helpers.
    state=100;mult=(1481765933<<32)|1284865837;mask=(1<<64)-1
    def values():
        nonlocal state
        for _ in range(n*(n+1)):
            state=(state*mult+1)&mask;lo=state&0xffffffff;hi=state>>32
            yield .5-(lo/4294967296.+hi)/4294967296.
    active=np.fromiter(values(),dtype=np.float64,count=n*(n+1)).reshape((n,n+1),order='F')
    total=np.zeros(ld*(n+1)+(n+1)+8,dtype=np.float64);total[:ld*(n+1)].reshape((ld,n+1),order='F')[:n]=active
    return total
ORIGINAL=[original_matrix(n,ld) for n,nb,ld in CASES]


def replay(path):
    a=np.memmap(path,dtype='<f8',mode='r');assert len(a)==27364678 and np.isfinite(a).all()
    words=np.asarray(a[:PREFIX]);assert np.equal(words,np.floor(words)).all() and np.all((words>=0)&(words<=4294967295))
    bits=words[::2].astype(np.uint64)|(words[1::2].astype(np.uint64)<<np.uint64(32));initial=bits.view(np.float64);offset=0
    for src in ORIGINAL:
        got=initial[offset:offset+len(src)];assert np.array_equal(got.view(np.uint64),src.view(np.uint64)), 'Independent sourceLCG/padding inputbit mismatch';offset+=len(src)
    assert offset*2==PREFIX
    pos=PREFIX;calls=[];max_ratio=0.;max_abs=0.;backwards=[];checked=0;first_control=None
    for ci,(n,nb,ld) in enumerate(CASES):
        assert tuple(a[pos:pos+3])==(n,nb,ld);pos+=3
        for call in range(12 if ci==0 else 0):
            h=a[pos:pos+9];assert np.equal(h[:7],np.floor(h[:7])).all();kind,m,nn,k,lda,ldb,ldc=map(int,h[:7]);alpha,beta=map(float,h[7:]);pos+=9
            assert kind in (0,1) and lda>=m and ldb>0 and ldc>0 and m>0 and nn>0
            def take(count):
                nonlocal pos
                z=a[pos:pos+count];assert len(z)==count;pos+=count;return z
            A=take(lda*(k if kind==0 else m)).reshape((lda,k if kind==0 else m),order='F')
            B=take(ldb*nn).reshape((ldb,nn),order='F')
            assert np.all(A[m:]==0),'ReadonlyA padding not normalized'
            if kind==0:
                assert np.all(B[k:]==0),'ReadonlyGEMMB padding not normalized'
                C=take(ldc*nn).reshape((ldc,nn),order='F');got=take(ldc*nn).reshape((ldc,nn),order='F')
                gold=alpha*(A[:m]@B[:k])+beta*C[:m]
                assert np.array_equal(got[m:].view(np.uint64),C[m:].view(np.uint64)),'GEMM preserved outputpadding changed'
            else:
                assert k==0 and lda>=m and ldb>=m
                got=take(ldb*nn).reshape((ldb,nn),order='F');gold=alpha*B[:m].copy()
                # Independent unit-lower forward substitution, not GPU/CPU trsm API.
                for row in range(m):
                    if row:gold[row]-=A[row,:row]@gold[:row]
                assert np.array_equal(got[m:].view(np.uint64),B[m:].view(np.uint64)),'DTRSM preserved outputpadding changed'
            errors=np.abs(got[:m]-gold);budget=1e-9+1e-9*np.abs(gold);ratio=float(np.max(errors/budget));assert ratio<=1,'Independent full operation output mismatch'
            max_ratio=max(max_ratio,ratio);max_abs=max(max_abs,float(np.max(errors)));checked+=got.size
            if first_control is None:
                first_control={'output_plus_one_rejected':bool(abs(float(got[0,0]+1-gold[0,0]))>float(budget[0,0])),'input_single_bit_change_rejected':bool((int(bits[0])^1)!=int(ORIGINAL[0].view(np.uint64)[0]))}
                assert all(first_control.values())
            calls.append({'kind':'DGEMM' if kind==0 else 'DTRSM','m':m,'n':nn,'k':k,'lda':lda,'ldb':ldb,'ldc':ldc,'max_normalized_error':ratio})
        count=ld*(n+1)+(n+1)+8;final=a[pos:pos+count];pos+=count;x=a[pos:pos+n+8];pos+=n+8;reported=float(a[pos]);pos+=1
        assert len(final)==count and np.all(final[-8:]==0) and np.all(x[n:]==987654.25),'Exact matrix/workspace/solution outerguard mismatch'
        M=ORIGINAL[ci][:ld*(n+1)].reshape((ld,n+1),order='F');AA=M[:n,:n];b=M[:n,n];sol=x[:n]
        # Extended CPU arithmetic for every dot; accuracy identity recorded, not universal highprecision.
        residual=np.asarray(AA,dtype=np.longdouble)@np.asarray(sol,dtype=np.longdouble)-np.asarray(b,dtype=np.longdouble)
        denom=float(np.max(np.sum(np.abs(AA),axis=1)))*float(np.max(np.abs(sol)))+float(np.max(np.abs(b)))
        backward=float(np.max(np.abs(residual)))/denom;assert backward<=1e-12
        backwards.append({'n':n,'recomputed_longdouble_backward_error':backward,'source_reported_backward_error':reported})
    assert pos==len(a) and len(calls)==12 and sum(c['kind']=='DTRSM' for c in calls)==1
    return {'pass':True,'artifact_fields':len(a),'exact_initial_word_fields':PREFIX,'GPU_call_geometry':calls,'full_stage_storage_checks':checked,'max_abs_operation_error':max_abs,'max_normalized_operation_error':max_ratio,'full_Ax_minus_b':backwards,'negative_controls':first_control,'guards':'Capturedmatrix/workspaceouter8 andsolutiontail8 exact; no deviceallocationouterguard claim.'}

result=json.loads(args.result.read_text());assert result['matched_performance_valid'] and result['reference_agreement']
cache={};records=[]
for model in ('cuda','cuda_jl','ka','jacc'):
    m=result['models'][model];assert m['status']=='native_check_pass' and m['complete'] and m['timing_valid'];rows=m['trials'];assert [t['trial'] for t in rows]==[1,2,3,4,5]
    assert abs(statistics.mean(t['time_seconds'] for t in rows[1:])-m['mean_seconds'])<1e-15
    for row in rows:
        art=row['output_artifact'];p=pathlib.Path(art['saved_path'])
        if args.run_root:p=args.run_root/p.parent.name/p.name
        digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==art['sha256'];assert p.stat().st_size==art['count']*8
        reuse=digest in cache
        if not reuse:cache[digest]=replay(p)
        records.append({'model':model,'trial':row['trial'],'path':str(p),'sha256':digest,'full_rehash_pass':True,'mathematical_replay_reused_identical_SHA':reuse})
out={'all_pass':True,'result':str(args.result),'result_sha256':hashlib.sha256(args.result.read_bytes()).hexdigest(),'artifacts_hashed':len(records),'unique_full_CPU_replays':len(cache),'all_models_exact_same_bits':len(cache)==1,'numpy_version':np.__version__,'longdouble_precision_bits':int(np.finfo(np.longdouble).nmant+1),'scope':'Recompute allinputbits fromindependentPythonLCG, fullGPUstage math withNumPyGEMM/explicitforwardsubstitution, everyrow longdoubleAx-b, exactouterguards andfive/meanlast4. Fullmathematical replay cachedonlyaftercompletefileSHAidentity; actualinprocessCPUchecks separatelyexecuted everytrial. No GPUrun.','records':records,'replays':cache}
args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(args.output);print(json.dumps({'all_pass':True,'artifacts':len(records),'unique_CPU_replays':len(cache),'max_ratio':max(v['max_normalized_operation_error'] for v in cache.values())}))
