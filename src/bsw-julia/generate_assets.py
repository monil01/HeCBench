#!/usr/bin/env python3
"""Deterministic named safe FASTA inputs and independent full-matrix oracle."""
import importlib.util,json,pathlib,subprocess,sys,array,hashlib
p=pathlib.Path(__file__).resolve().parent
subprocess.run([sys.executable,str(p/'native_diagnostics/generate_readiness_fasta.py'),str(p)],check=True)
s=importlib.util.spec_from_file_location('sw_oracle',p/'native_diagnostics/audit_cli.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
cases=json.loads((p/'fixture.json').read_text())['cases'];N=len(cases);aa=[c[0]for c in cases];bb=[c[1]for c in cases];sa=list(''.join(aa).encode());sb=list(''.join(bb).encode());oa=[];ob=[];a=b=0
outputs=[[]for _ in range(5)];ties=[]
for x,y in cases:
 a+=len(x);b+=len(y);oa.append(a);ob.append(b);score,(ae,be),t=o.sw(x,y);back,(ac,bc),rt=o.sw(x[:ae][::-1],y[:be][::-1],True);assert score==back
 for v,z in zip(outputs,[ae-ac,ae,be-bc,be,score]):v.append(z)
 ties.append([t,rt])
maxa=max(map(len,aa));maxb=max(map(len,bb));fwd=min(maxa,maxb);rev=min(max(outputs[1]),max(outputs[3]));sh=3*(fwd+1)*2;sh+=4+(4-sh%4)
wrap=lambda v,g:[g]*16+v+[g]*16
before=wrap([0]*len(sa),211)+wrap([0]*len(sb),211)+wrap([0]*N,24681357)+wrap([0]*N,24681357)+wrap(o.mat,-12345)+wrap(o.enc,-12345)
after=wrap(sa,211)+wrap(sb,211)+wrap(oa,24681357)+wrap(ob,24681357)+wrap(o.mat,-12345)+wrap(o.enc,-12345)
init=sum([wrap([-12345]*N,-12345)for _ in range(5)],[]);out=sum([wrap(v,-12345)for v in outputs],[])
prefix=[N,maxa,maxb,-6,-1,fwd,rev]+before+after+outputs[1]+outputs[3]+init;expected=prefix+out
(p/'oracle_i32.bin').write_bytes(array.array('i',expected).tobytes())
header='#pragma once\n#include <cstdint>\n'
for key,v in [('N',N),('NA',len(sa)),('NB',len(sb)),('FORWARD',fwd),('REVERSE',rev),('SHMEM',sh),('PREFIX',len(prefix)),('COUNT',len(expected))]:header+=f'constexpr int {key}={v};\n'
for key,v,typ in [('SEQ_A',sa,'unsigned char'),('SEQ_B',sb,'unsigned char'),('OFF_A',oa,'unsigned'),('OFF_B',ob,'unsigned'),('SCORING',o.mat,'short'),('ENCODING',o.enc,'short')]:header+=f'const {typ} {key}[]={{'+','.join(map(str,v))+'};\n'
(p/'study_params.h').write_text(header)
jl='\n'.join(f'const {k}={v}'for k,v in [('N',N),('NA',len(sa)),('NB',len(sb)),('FORWARD',fwd),('REVERSE',rev),('SHMEM',sh),('PREFIX',len(prefix)),('COUNT',len(expected))])+'\n'
for key,v,typ in [('SEQ_A',sa,'UInt8'),('SEQ_B',sb,'UInt8'),('OFF_A',oa,'UInt32'),('OFF_B',ob,'UInt32'),('SCORING',o.mat,'Int16'),('ENCODING',o.enc,'Int16')]:jl+=f'const {key}={typ}['+','.join(map(str,v))+']\n'
(p/'study_params.jl').write_text(jl)
plan=dict(benchmark='bsw',cases=len(cases),sequence_bytes=[len(sa),len(sb)],max_lengths=[maxa,maxb],threads_forward=fwd,threads_reverse=rev,dynamic_shared_bytes=sh,source_gap_open=-6,source_gap_extend=-1,score_encoding='BLOSUM62 from unchangedsource table',precision='Int16 sourceDP/score; bounds maxpossible score <=11*minlength<32767',input_prefix_count=len(prefix),artifact_count=len(expected),oracle='Independent row-major full-matrix affine-gapDP plus declared source endpoint tie ordering; source kernel antidiagonal/shuffle algorithm is not used as CPUoracle.',endpoint_order='forwardmin(longerrow,shortercol); reversemin(shortercol,longerrow) in DPconsumedcoordinates',tie_maxima=ties,scope='Named65positivepairs, sourceits1 batch, source hostpacking/H2D/forward/endreadback/hostnewMin/reverse/outputreadback measured; preallocation/constantsoutsideROI, diagnosticprintfexcludedall4. OriginalDVCdataset/>20kbatching/zero-score/FASTAlinewrappingunmeasured. maxref>=maxquery prevents sourcehostBallocationoverrun on thisfixture.',generator_python=sys.version,files={name:dict(bytes=(p/name).stat().st_size,sha256=hashlib.sha256((p/name).read_bytes()).hexdigest())for name in ['reference.fasta','query.fasta','fixture.json','oracle_i32.bin','study_params.h','study_params.jl']})
(p/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({k:v for k,v in plan.items()if k in ['artifact_count','input_prefix_count','threads_forward','threads_reverse','dynamic_shared_bytes']}))
