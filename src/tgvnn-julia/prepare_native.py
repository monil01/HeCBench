#!/usr/bin/env python3
"""Source-bound host instrumentation; device function bodies remain original include."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;R=P.parents[1];s=(R/'src/tgvnn-cuda/src/main.cu').read_text();start=s.index('void\ntgv_cs(');end=s.index('\nint\nmain (',start);body=s[start:end].replace('tgv_cs(', 'observed_tgv_cs(',1)
# Original helper/device definitions remain in the unchanged source include.
repls=[('  // Scale the data to [0,1]', '  observe_inputs(d_img,d_mask,N);\n\n  // Scale the data to [0,1]'),
 ('    // update q','    if (i==3) observe_trace("p",d_p,3*N);\n\n    // update q'),
 ('    // update p','    if (i==3) observe_trace("r",d_r,N);\n\n    // update p'),
 ('    // update s','    if (i==3) observe_trace("q",d_q,6*N);\n\n    // update s'),
 ('    // update w','    if (i==3) observe_trace("s",d_imgs,N);\n\n    // update w'),
 ('    // update l','    if (i==3) observe_trace("w",d_w,3*N);\n\n    // update l'),
 ('    // update intermediate variables','    if (i==3) observe_trace("l",d_imgl,N);\n\n    // update intermediate variables'),
 ('    arrayadd<<<gridsize, blocksize>>>(d_lbar, d_imgl, d_lold, N, 2.f, -1.f);','    arrayadd<<<gridsize, blocksize>>>(d_lbar, d_imgl, d_lold, N, 2.f, -1.f);\n    if(i==3){ observe_trace("lb",d_lbar,N);observe_trace("sb",d_sbar,N);observe_trace("wb",d_wbar,3*N);observe_trace("tmp",d_tmp,12*N); }'),
 ('  // Free arrays and destroy handles','  observe_final(N,lds,{'+','.join('{"'+k+'",'+v+','+n+'}'for k,v,n in [('img','d_img','N'),('mask','d_mask','N'),('imgz','d_imgz','N'),('imgb','d_imgb','N'),('imgl','d_imgl','N'),('imgs','d_imgs','N'),('w','d_w','3*N'),('p','d_p','3*N'),('q','d_q','6*N'),('r','d_r','N'),('lbar','d_lbar','N'),('sbar','d_sbar','N'),('wbar','d_wbar','3*N'),('lold','d_lold','N'),('sold','d_sold','N'),('wold','d_wold','3*N'),('tmp','d_tmp','12*N'),('imgr','d_imgr','N')])+ '},d_ls,d_ls2,d_lu,d_lvt,d_lsvt,devInfo,ldu,ldvt,ser_imgz,ser_imgr);\n\n  // Free arrays and destroy handles')]
for old,new in repls:
 assert body.count(old)==1,(old,body.count(old));body=body.replace(old,new)
body='\n'.join(l for l in body.splitlines()if 'printf('not in l and 'printf ('not in l)+'\n'
out=P/'observed_tgv.inc';out.write_text(body)
proof={'original_source':str((R/'src/tgvnn-cuda/src/main.cu').relative_to(R)),'original_sha256':hashlib.sha256(s.encode()).hexdigest(),'extracted_original_function_sha256':hashlib.sha256(s[start:end].encode()).hexdigest(),'instrumented_host_function':str(out.relative_to(R)),'instrumented_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'changes':[x[0]for x in repls]+['rename host tgv_cs to observed_tgv_cs','omit source Running/SER printf formatting from controlled timed ROI'],'GPU_function_bodies_modified':False,'math_modified':False,'execution_adapted':True,'no_original_CLI_fidelity_claim_from_instrumented_adapter':True};(P/'native_diagnostics/host_instrumentation_provenance.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
