"""CPU-only full integer-COO oracle and all20 complete actual capture/source audit."""
import argparse,json,hashlib,math,pathlib,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--fixtures',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();start=time.monotonic();result=json.load(open(a.result));assert result['matched_performance_valid'];cases=[]
for name,R in [('toy',32),('control3D',7),('longFiber4D',3),('sourceSize4D',32)]:
 f=a.fixtures/(name+'.tns')
 with f.open()as stream:nd=int(next(stream));dims=list(map(int,next(stream).split()))
 data=np.loadtxt(f,skiprows=2,ndmin=2);assert data.shape[1]==nd+1;ix=data[:,:nd].astype(np.int64)-1;assert np.array_equal(ix+1,data[:,:nd]);vals=data[:,-1].astype(np.float32);bits=vals.view(np.uint32);exps=((bits>>23)&255).astype(np.int64);assert np.all(exps!=255);mant=(bits&0x7fffff).astype(np.int64)+np.where(exps!=0,0x800000,0);mant=np.where(bits>>31,-mant,mant);gold=[]
 for mode,d in enumerate(dims):
  assert np.all((ix[:,mode]>=0)&(ix[:,mode]<d));bins=np.zeros((d,256),dtype=np.int64);np.add.at(bins,(ix[:,mode],exps),mant);odd=math.prod(2*m+1 for m in range(nd)if m!=mode)
  exact=[]
  for row in bins:
   units=sum(int(v)<<(e-1 if e else 0)for e,v in enumerate(row)if v);exact.append(math.ldexp(float(units*odd),-149-(nd-1)))
  gold.append(np.repeat(np.array(exact),R))
 cases.append(dict(name=name,nd=nd,dims=dims,R=R,nz=len(data),gold=gold,fixture_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 del data,ix,vals,bits,exps,mant,bins
artifact_rows=[];dependency_rows=[];cache={};overall_gold=overall_double=overall_projected=0.
for model,record in result['models'].items():
 assert record['complete']and record['provenance_valid']and record['timing_valid'];assert len(record['trials'])==5;assert math.isclose(record['mean_seconds'],sum(x['time_seconds']for x in record['trials'][1:])/4,rel_tol=1e-15)
 for path,h in record['adapter_file_hashes'].items():
  if path not in cache:cache[path]=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
  assert cache[path]==h,(model,path);dependency_rows.append(dict(model=model,path=path,sha256=h))
 for trial in record['trials']:
  assert trial['correct'];art=trial['output_artifact'];f=pathlib.Path(art['saved_path']);sha=hashlib.sha256(f.read_bytes()).hexdigest();assert sha==art['sha256'];v=np.fromfile(f,dtype='<f8');assert len(v)==4514748 and np.all(np.isfinite(v));pos=0;pre=[]
  for ci,c in enumerate(cases):
   meta=v[pos:pos+7];pos+=7;assert list(meta[:4])==[ci,c['nd'],c['nz'],c['R']];assert list(v[pos:pos+c['nd']])==c['dims'];pos+=c['nd'];offset=np.r_[0,np.cumsum(np.array(c['dims'])*c['R'])];size=int(offset[-1]);inputs=[]
   for mode in range(c['nd']):
    n=c['dims'][mode]*c['R'];init=v[pos:pos+n+128];pos+=n+128;assert np.all(init[:64]==1234567.25)and np.all(init[-64:]==1234567.25)and np.all(init[64:-64]==0);assert list(v[pos:pos+2])==[ci,mode];pos+=2;before=v[pos:pos+size];pos+=size
    for m in range(c['nd']):assert np.all(before[offset[m]:offset[m+1]]==(0 if m==mode else m+.5))
    inputs.append(before)
   pre.append((offset,inputs))
  assert pos==result['input_prefix_count']==2052307;md=mp=mg=0.
  for ci,c in enumerate(cases):
   offsets,inputs=pre[ci];size=int(offsets[-1])
   for mode in range(c['nd']):
    n=c['dims'][mode]*c['R'];double=v[pos:pos+n+128];pos+=n+128;assert np.all(double[:64]==1234567.25)and np.all(double[-64:]==1234567.25);assert list(v[pos:pos+2])==[ci,mode];pos+=2;got=v[pos:pos+size];pos+=size;stored=v[pos:pos+n];pos+=n;gold=c['gold'][mode];budget=.01+2e-5*np.abs(gold);d=np.abs(double[64:-64]-gold);e=np.abs(got[offsets[mode]:offsets[mode+1]]-gold);g=np.abs(stored-gold);assert np.all(d<=budget)and np.all(e<=budget);assert np.all(g==0);assert np.array_equal(double[64:-64].astype(np.float32).astype(np.float64),got[offsets[mode]:offsets[mode+1]])
    for m in range(c['nd']):
     if m!=mode:assert np.array_equal(got[offsets[m]:offsets[m+1]],inputs[mode][offsets[m]:offsets[m+1]])
    md=max(md,float(d.max()));mp=max(mp,float(e.max()));mg=max(mg,float(g.max()))
  assert pos==len(v);overall_double=max(overall_double,md);overall_projected=max(overall_projected,mp);overall_gold=max(overall_gold,mg);artifact_rows.append(dict(model=model,trial=trial['trial'],path=str(f),sha256=sha,count=len(v),max_FP64_abs_error=md,max_projected_FP32_abs_error=mp,max_stored_gold_vs_exact_integer_oracle=mg,all_full_guards_inputs_readonly_projection_checks_pass=True))
out=dict(benchmark='mmcsf',configuration_id=result['configuration_id'],result=str(a.result),result_sha256=hashlib.sha256(a.result.read_bytes()).hexdigest(),status='pass',method='All actual Float32 textinputs converted to signed exact integer units2^-149, summedperrow/permode via bounded integer significand bins then arbitraryprecisioninteger superaccumulation. Exact source constantpremodefactors independentlyproved for every actualfield; complete oddinteger/poweroftwo contraction, allcolumns/trials. No GPU/no referencevalue fitting.',cases=[{k:v for k,v in c.items()if k!='gold'}for c in cases],artifacts=artifact_rows,source_dependencies=dependency_rows,unique_source_paths=len(cache),full_saved_artifact_count=len(artifact_rows),max_FP64_abs_error=overall_double,max_projected_FP32_abs_error=overall_projected,max_stored_CPUgold_vs_exact_integer_oracle=overall_gold,elapsed_seconds=time.monotonic()-start,primary_original_strict_contract='remains blocked; no promotion or tolerance change',scope='Qualified mixedprecisionsecondary only. Canonical actual full CSFreadbacks retained via exact byte-identical dedup/runtimeall-byte comparisons; sourcecanonical readback equality itself is audited by C++contract and hash bindings, not reexecuted here. Full source original CLI/generalranks/CPsolver not certified.')
q=a.output.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(a.output);print(json.dumps({k:out[k]for k in['status','full_saved_artifact_count','unique_source_paths','max_FP64_abs_error','max_projected_FP32_abs_error','max_stored_CPUgold_vs_exact_integer_oracle','elapsed_seconds']}))
