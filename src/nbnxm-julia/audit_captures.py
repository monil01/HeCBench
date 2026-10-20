#!/usr/bin/env python3
"""Full saved-output CPU replay using independent Float64 analytic pair math.
Canonical and output files are fully rehashed; no GPU calls or source-code changes.
"""
import pathlib,argparse,json,hashlib,array,struct,math,sys
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert sys.byteorder=='little';B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent;r=json.loads(a.result.read_text());cfg=json.loads((a.result.parent/'config.json').read_text());assets=cfg['input_contract']['actual_input_readback_assets'];names=['originalNoShift','originalShift','centralBipartite','shiftedBipartiteNoShift','shiftedBipartiteShift','centralSelf'];cases=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def corr(x):
 if x<.01:
  t=1.;s=0.
  for k in range(40):s+=t/(2*k+3);t*=-x/(k+1)
  return -4/math.sqrt(math.pi)*s
 z=math.sqrt(x);return 2*math.exp(-x)/(math.sqrt(math.pi)*x)-math.erf(z)/(x*z)
for caseID,name in enumerate(names):
 asset=next(x for x in assets if x['path'].endswith('actual_input_'+name+'.bin'));f=R/asset['path'];actualsha=sha(f);assert actualsha==asset['sha256'];raw=f.read_bytes();magic=struct.unpack_from('<Q',raw)[0];assert magic==0x4e42494e50555431;head=struct.unpack_from('<8i4f',raw,8);n,NCJ,NSCI,NEX,nt,gz,repeat,flag=head[:8];cut,beta,eps,padding=head[8:];assert struct.unpack_from('<Q',raw,56)[0]==9;off=64;data=[]
 for kind in ['f','f','f','f','I','i','I','i','f']:
  size=struct.unpack_from('<Q',raw,off)[0];off+=8;b=array.array(kind);b.frombytes(raw[off:off+size]);off+=size;data.append(b)
 assert off==len(raw);xq,initialF,sh,initialS,cj,sci,ex,types,coeff=data;assert[len(v)for v in data]==[4*n,3*n,135,135,8*NCJ,4*NSCI,32*NEX,n,2*nt*nt];goldF=array.array('d',initialF);goldS=array.array('d',initialS);touchedF=[False]*n;touchedS=[False]*45;pairs=excluded=cutoff=selfRejected=0
 for block in range(gz):
  sc,shift,start,end=sci[4*block:4*block+4];assert 0<=shift<45 and 0<=start<=end<=NCJ
  for j4 in range(start,end):
   for j in range(8):
    warp=j//4;mask=cj[8*j4+4+2*warp];ei=cj[8*j4+5+2*warp];assert ei<NEX
    if not mask:continue
    for jm in range(4):
     for ciOffset in range(8):
      bit=1<<(jm*8+ciOffset)
      if not mask&bit:continue
      ci=sc*8+ciOffset;jcluster=cj[8*j4+jm];aj=jcluster*8+j
      for i in range(8):
       ai=ci*8+i;assert 0<=ai<n and 0<=aj<n
       if shift==22 and j<=i and ci==jcluster:selfRejected+=1;continue
       rv=[float(xq[4*ai+k])+sh[3*shift+k]-xq[4*aj+k]for k in range(3)];r2=sum(x*x for x in rv)
       if not r2<cut:cutoff+=1;continue
       r2=max(r2,struct.unpack('<f',struct.pack('<f',3.82e-7))[0]);inv=1/math.sqrt(r2);inv2=1/r2;inv6=inv2**3;included=bool(ex[32*ei+((8*j+i)&31)]&bit);ti,tj=types[ai],types[aj];assert 0<=ti<nt and 0<=tj<nt;c6,c12=coeff[2*(nt*ti+tj):2*(nt*ti+tj)+2];lj=inv6*(c12*inv6-c6)*inv2 if included else 0.;qq=eps*xq[4*ai+3]*xq[4*aj+3];elec=qq*((inv2*inv if included else 0)+beta**3*corr(beta*beta*r2));scale=lj+elec
       for k in range(3):
        force=rv[k]*scale;goldF[3*ai+k]+=force;goldF[3*aj+k]-=force
        if flag and shift!=22:goldS[3*shift+k]+=force
       touchedF[ai]=touchedF[aj]=True
       if flag and shift!=22:touchedS[shift]=True
       pairs+=1;excluded+=not included
 meta=array.array('d',[caseID,*head[:8],*head[8:11]]);prefix=meta+array.array('d',initialF)+array.array('d',initialS);cases.append(dict(name=name,header=head,n=n,pairs=pairs,excluded=excluded,cutoff=cutoff,selfRejected=selfRejected,initialF=initialF,initialS=initialS,goldF=goldF,goldS=goldS,touchedF=touchedF,touchedS=touchedS,prefix=prefix,canonical_sha256=actualsha))
expectedPrefix=array.array('d')
for c in cases:expectedPrefix.extend(c['prefix'])
assert len(expectedPrefix)==1231362;prefixBytes=expectedPrefix.tobytes();rows=[]
for model,m in r['models'].items():
 if not m.get('complete'):continue
 for t in m['trials']:
  f=pathlib.Path(t['output_artifact']['saved_path']);assert f.stat().st_size==2462658*8;actualsha=sha(f);assert actualsha==t['output_artifact']['sha256'];v=array.array('d');v.frombytes(f.read_bytes());assert v[:len(expectedPrefix)].tobytes()==prefixBytes;i=len(expectedPrefix);summaries=[]
  for caseID,c in enumerate(cases):
   assert v[i]==caseID;i+=1;outF=v[i:i+3*c['n']];i+=3*c['n'];outS=v[i:i+135];i+=135;mx=ratio=0.;untouched=True
   for got,gold,initial,touched in [(outF,c['goldF'],c['initialF'],c['touchedF']),(outS,c['goldS'],c['initialS'],c['touchedS'])]:
    if c['pairs']==0:
     assert got.tobytes()==array.array('d',initial).tobytes()
    else:
     for j,(x,y,z)in enumerate(zip(got,gold,initial)):
      assert math.isfinite(x);delta=abs(x-y);budget=.002+1e-4*abs(y);mx=max(mx,delta);ratio=max(ratio,delta/budget);assert delta<=budget
      if not touched[j//3]:assert x==z
   net=[math.fsum(outF[j]-c['initialF'][j]for j in range(k,len(outF),3))for k in range(3)];magnitude=math.fsum(abs(x-y)for x,y in zip(c['goldF'],c['initialF']));newton=max(map(abs,net))/max(1.,magnitude);assert newton<=5e-5
   summaries.append({'name':c['name'],'decoded_active_pairs':c['pairs'],'max_independent_abs_error':mx,'max_error_over_budget':ratio,'normalized_Newton_residual':newton,'exact_input_prefix_and_untouched_regions_pass':True,'default_zero_mask_bit_identity_pass':True if c['pairs']==0 else None})
  assert i==len(v);rows.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':actualsha,'checked_values':len(v),'cases':summaries,'pass':True})
out={'benchmark':'nbnxm','result_path':str(a.result),'result_sha256':sha(a.result),'canonical_assets_fully_rehashed':len(assets),'full_output_artifacts_rehashed':len(rows),'all_pass':True,'oracle':'SeparatePythonFloat64 analyticpairenumerationfromfullactualnativecanonicalbuffers; fullbytesexactbaseline/untouchedfields and everyactiveforce/fShift field. NotreusingnativeC++forceoraclecode; analyticalmathematicalmodelshared.','scope':'CompletefieldCPUreplayofsavedacceptedcaptures, noGPUrerun. Originaldefaulthaszeroactivepairs; namedcontrolsassurephysicalforcecodecoverage. PaircountsdecodedCPUvaluesnotGPUhardwarecounters.','artifacts':rows};q=a.output.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(a.output);print(json.dumps({k:v for k,v in out.items()if k not in ['artifacts']}))
