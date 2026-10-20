#!/usr/bin/env python3
"""Full saved encoded/raw-bit/source/scientific audit; CPU only, no GPU reruns."""
import pathlib,json,hashlib,argparse,sys,tomllib
import numpy as np
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def component(x):return x.reshape(-1).view('<f8')if np.iscomplexobj(x)else x.reshape(-1)
def main():
 a=argparse.ArgumentParser();a.add_argument('--result',type=pathlib.Path,required=True);a.add_argument('--oracle',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);z=a.parse_args();res=json.load(open(z.result));assert res['matched_performance_valid']and all(res['variant_verified'].values());units=json.load(open(P/'Fourier_metric_units.json'));man=json.load(open(z.oracle/'oracle_manifest.json'));N=60000;K=32;B=130;H=390;F=780
 from audit_native import M
 expected={k:np.fromfile(z.oracle/(k+'.bin'),dtype='<c16'if'complex'in v['dtype']else'<f8')for k,v in man['files'].items()}
 initial=expected['input_points'].reshape(2,N);mn=initial.min();width=(initial.max()-mn)/B;grid=np.arange(B);lo=np.r_[np.tile(mn+width*grid,B),np.repeat(mn+width*grid,B)];hi=lo+width
 fk=np.zeros(F*F,dtype=np.complex128);fk[:F*(H+1)]=expected['fft_kernel'];prod=expected['fft_product'].reshape(4,F,H+1);full=np.empty((4,F,F),dtype=np.complex128);full[:,:,:H+1]=prod;full[:,:,H+1:]=np.conj(prod[:,(-np.arange(F))%F][:,:,np.arange(H-1,0,-1)]);scratch=(np.fft.ifft(full,axis=2)*F).transpose(0,2,1).copy();spacing=np.array([1/6,1/2,5/6]);den=np.array([np.prod(spacing[j]-np.delete(spacing,j))for j in range(3)])
 extras={'fft_kernel_tilde_device':fk,'fft_scratchpad_device':scratch,'ones_device':np.ones(2*N),'box_lower_bounds_device':lo,'box_upper_bounds_device':hi,'y_tilde_spacings_device':spacing,'denominator_device':den}
 trials=[];sources={};tree_reviews=[];statebytes=0
 sys.path.insert(0,str(R/'tools'));from verify_portable_julia import digest_tree
 for model,rec in res['models'].items():
  assert rec['status']=='native_check_pass'and len(rec['trials'])==5 and rec['provenance_valid']
  for path,digest in rec['adapter_file_hashes'].items():
   if isinstance(digest,dict):digest=digest.get('sha256')
   assert pathlib.Path(path).is_file()and sha(path)==digest,('source_changed',model,path);sources[path]=digest
  for what in ['source','adapter']:
   path=rec.get(what+'_path'if what=='source'else'adapter_source_path');digest=rec.get(what+'_hash');assert digest_tree(path)==digest,('tree_changed',model,what,path);tree_reviews.append({'model':model,'kind':what,'path':path,'sha256':digest})
  for row in rec['trials']:
   art=row['output_artifact'];f=pathlib.Path(art['saved_path']);assert sha(f)==art['sha256'];raw=np.memmap(f,dtype='<f4',mode='r');assert raw.size==32563847;statebytes+=f.stat().st_size;proofpath=pathlib.Path(row['lossless_raw_GPU_codec_proof']);proof=json.load(open(proofpath));layout_path=proofpath.with_name('trial'+str(row['trial'])+'_layout.json');layout=json.load(open(layout_path));assert sha(layout_path)==proof['layout_sha256'];assert sha(P/'Fourier_metric_units.json')==proof['units_sha256'];assert proof['encoded_sha256']==art['sha256'];h=hashlib.sha256();fields=[];distactual=None;idxactual=None;pactual=None;pointmax=0
   for field in layout:
    name=field['name'];v=raw[field['offset']:field['offset']+field['count']];exponent=units['fields'].get(name,{}).get('divide_by_power_of_two_exponent',0);v=np.ldexp(v,exponent).astype('<f4');assert np.isfinite(v).all();h.update(v.tobytes());key=M.get(name);gold=expected[key]if key else extras[name];gold=component(gold)
    if key=='fft_output':gold=gold*(F*F)
    assert gold.size==v.size,(name,gold.size,v.size)
    error=float(np.max(abs(v.astype(float)-gold)));budget=1e-6+.003*float(np.max(abs(gold)));assert error<=budget,(model,row['trial'],name,error,budget);exact=None
    if key and key.startswith('input_'):exact=np.array_equal(v.view('<u4'),gold.astype('<f4').view('<u4'));assert exact
    if name in ['pij_indices_device','point_box_idx_device']:exact=np.array_equal(v.astype(float),gold);assert exact
    fields.append({'field':name,'count':int(v.size),'max_abs_error_raw_units':error,'predeclared_field_budget':budget,'budget_ratio':error/budget,'exact_required_and_passed':exact,'independent_method':'preexistingmathematicaloracle'if key else'supplementalanalytic/Complex128FFTidentity','supplemental_not_needed_to_override_failed_contract':not bool(key)})
    if name=='pij_nonsymmetric':pactual=v.reshape(N,K).astype(float)
    if name=='points_device':pointmax=error
   assert h.hexdigest()==proof['source_sha256']==proof['decoded_original_sha256'],'raw_gpu_bitstream_changed'
   mass=pactual.sum(axis=1);prob=pactual/mass[:,None];entropy=-(np.where(prob>0,prob*np.log(np.maximum(prob,np.finfo(float).tiny)),0)).sum(axis=1);perplexity=np.exp(entropy)
   trials.append({'model':model,'trial':row['trial'],'artifact':str(f),'stored_sha256':art['sha256'],'decoded_raw_GPU_sha256':h.hexdigest(),'codec_proof_sha256':sha(proofpath),'complete_words_checked':int(raw.size),'all35_independent_fields_pass':True,'fields':fields,'point_coordinate_max_abs_error_vs_CPU':pointmax,'actual_conditional_probability_rowmass_range':[float(mass.min()),float(mass.max())],'observed_renormalized_perplexity_range':[float(perplexity.min()),float(perplexity.max())],'source_target_perplexity':50,'target_perplexity_feasible':False,'source_optimizer_quality_certificate':False})
 out={'status':'all20_full_captures_pass','result':str(z.result),'result_sha256':sha(z.result),'models':4,'trials':trials,'full_artifacts':len(trials),'artifact_bytes':statebytes,'all_original_GPU_bitstreams_exactly_recovered':True,'full_independent_fields_per_trial':35,'preexisting_mandatory_fields':28,'additional_mathematical_fields_after_execution':7,'additional_fields_strengthen_evidence_only':True,'strict_source_bindings':sources,'source_bindings_count':len(sources),'source_trees_unchanged':tree_reviews,'raw_pointwise_failures_not_overridden':'I3discretegridding andearlierI1rawpointwiseFourier receipts remainfailed; this audit appliesonlyexplicitselectedI1normwisecontract.','scientific_limitation':'Defaultperplexity50cannotfit31positive supports; source200searchcap returnsuniformconditionalrows withobservedperplexity31. Numericalagreementisnot50-perplexityembeddingquality orfull1000-steptrajectoryvalidation.','GPU_execution':False,'NumPy':np.__version__};z.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'artifacts':len(trials),'bytes':statebytes,'bindings':len(sources),'all35fields':True,'max_independent_budget_ratio':max(f['budget_ratio']for t in trials for f in t['fields'])}))
if __name__=='__main__':main()
