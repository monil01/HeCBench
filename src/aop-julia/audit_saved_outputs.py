from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/home/imo/HeCBench_julia');B=R/'src/aop-julia';S=R/'.porting-state/portable_julia';P=next(p for p in reversed(sorted((S/'runs').glob('*_aop/result.json')))if json.loads(p.read_text()).get('matched_performance_valid'));d=json.loads(P.read_text());cs=json.loads((B/'cases.json').read_text())['cases'];f=np.memmap(B/'input_f64.bin',dtype='<f8',mode='r');o=np.memmap(B/'oracle_paths_f64.bin',dtype='<f8',mode='r');G=16;GUARD=-987654321.25;POISON=-1234567.25;rows=[]
for model,m in d['models'].items():
 for tr in m['trials']:
  ar=tr['output_artifact'];v=np.memmap(ar['saved_path'],dtype='<f8',mode='r');off=0;initial=[];cases=[]
  for c in cs:
   fields=[]
   for k,n in enumerate([c['n']*c['steps'],c['n']*c['steps'],16*c['steps'],8192,c['steps']]):
    a=v[off:off+n+32];off+=n+32;g=-55555 if k==4 else GUARD;assert np.all(a[:16]==g)and np.all(a[-16:]==g)
    if k==0:assert np.array_equal(a[16:-16],f[c['io']:c['io']+n])
    else:assert np.all(a[16:-16]==(-55555 if k==4 else POISON))
    fields.append(a)
   initial.append(fields)
  assert off==6653220
  for c,pre in zip(cs,initial):
   fields=[]
   for k,n in enumerate([c['n']*c['steps'],c['n']*c['steps'],16*c['steps'],8192,c['steps']]):
    a=v[off:off+n+32];off+=n+32;g=-55555 if k==4 else GUARD;assert np.all(a[:16]==g)and np.all(a[-16:]==g);fields.append(a[16:-16])
    if k==0:assert np.array_equal(a,pre[k])
   samples,paths,svds,temp,flags=fields;n=c['n'];steps=c['steps'];paths=paths.reshape(steps,n);svds=svds.reshape(steps,16);want=o[c['oo']:c['oo']+n*steps].reshape(steps,n);e=float(np.abs(paths-want).max());assert np.all(np.abs(paths-want)<=2e-6+1e-5*np.abs(want));assert np.all(svds[:,12:]==POISON)and np.all(svds[-1]==POISON);assert flags[-1]==0
   normalerr=0.;inverr=0.;projerr=0.;maxcond=0.;ranks=[]
   for t in range(steps-1):
    x=paths[t];inside=x<c['K']if c['put']else x>c['K'];mcount=int(inside.sum());assert flags[t]==int(mcount<4)
    if mcount<4:
     assert svds[t,0]==POISON
     selected=x[inside][:3];expected=np.zeros(12);expected[:len(selected)]=selected
     assert np.array_equal(svds[t,1:12],expected[1:]);continue
    a=np.array([np.ones(mcount),x[inside],x[inside]**2]).T
    rr=np.zeros((3,3));rr[np.triu_indices(3)]=svds[t,:6];ww=np.zeros((3,3));ww[np.triu_indices(3)]=svds[t,6:12]
    gram=a.T@a;ng=float(np.max(np.abs(rr.T@rr-gram))/max(1.,np.max(np.abs(gram))));normalerr=max(normalerr,ng);assert ng<2e-10
    u,s,vh=np.linalg.svd(rr);keep=s*s>=1e-12;expected=(vh.T*np.where(keep,1/s,0.))@u.T;ee=float(np.linalg.norm(ww-expected)/max(1.,np.linalg.norm(expected)));inverr=max(inverr,ee);
    if ee>=2e-5:print("INVERSE_DIAGNOSTIC",model,tr["trial"],c["case_id"],t,"rank",int(keep.sum()),"relerror",ee,"R",rr.tolist(),"W",ww.tolist(),"pinvR",expected.tolist(),flush=True)
    if int(keep.sum())==3:assert ee<2e-5
    pp=ww@rr;pe=float(np.linalg.norm(pp@pp-pp)/max(1.,np.linalg.norm(pp)));projerr=max(projerr,pe);
    if pe>=2e-5:print("PROJECTOR_DIAGNOSTIC",model,tr["trial"],c["case_id"],t,"rank",int(keep.sum()),"error",pe,flush=True)
    if int(keep.sum())==3:assert pe<2e-5;ranks.append(int(keep.sum()));maxcond=max(maxcond,float(s[0]/s[-1])if s[-1]>0 else 0.)
   # Original final reduction writes firstceil(n/128)temp slots; previous beta partials may write first384.
   written=max(384 if np.any(flags[:-1]==0) else 3,(n+127)//128);assert np.all(temp[written:]==POISON)
   price=np.exp(-c['r']*c['dt'])*paths[-1].sum()/n;assert abs(price-temp[0])<2e-12 and abs(price-tr['prices'][len(cases)])<2e-12;assert np.all(paths[-1]>=0)
   cases.append(dict(n=n,steps=steps,max_full_path_cashflow_abs_error=e,relative_normal_Gram_residual=normalerr,relative_inverse_R_disagreement=inverr,relative_retained_subspace_idempotence=projerr,max_R_condition=maxcond,retained_ranks=sorted(set(ranks)),all_actual_inputs_unchanged=True,all_legal_flags_exact=True,all_unwritten_SVD_and_temp_slots_exact=True,all_guards_exact=True,price_vs_discounted_mean_abs_error=float(abs(price-temp[0]))))
  assert off==len(v)
  with Path(ar['saved_path']).open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==ar['sha256']
  rows.append(dict(model=model,trial=tr['trial'],sha256=ar['sha256'],cases=cases))
  del v
(B/'output_audit.json').write_text(json.dumps(dict(result=str(P.relative_to(R)),result_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),checked_artifacts=len(rows),artifact_words=13306440,input_prefix_words=6653220,raw_SHA_classes=len({r['sha256']for r in rows}),all_declared_translation_checks_pass=True,scientific_accuracy_all_regimes_pass=False,independent_full_rank_diagnostics_pass=True,rank_one_pseudoinverse_diagnostics_pass=False,scope='Full actual GPU final path/cashflow, sourceQR/Wworkspaces/flags/temp/input/guards. Independent stableSVD oracle and alltimesteps actualR normalGram/pseudoinverse/subspace tests. Finalprice discountedmean. No actual per-backward-step GPUdecision/betatrace; no continuousAmericanpricingaccuracy or t0exercise certificate.',trials=rows),indent=2)+'\n');print('all20 source-agreement/input/guard/full-rank checks pass; rank-one pseudoinverse diagnostics FAIL as documented',len(rows))
