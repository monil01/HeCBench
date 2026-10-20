"""CPU-only independent check of all saved head/tail bytes and guard scope."""
from pathlib import Path
import json,argparse,hashlib,re,math
S=Path(__file__).resolve().parent;P=S.parent;R=P.parents[1]
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.load(open(a.result));assert d['matched_performance_valid'];inventory=[]
for f in [P/'observed_fit.inc',*sorted((R/'src/tsne-cuda').glob('*.cu'))]:
 if f.name in['main.cu','fit_tsne.cu']:continue
 for typ,name in re.findall(r'thrust::device_vector<([^;\n]+?)>\s+([A-Za-z_]\w*)\s*\(',f.read_text()):inventory.append({'source':str(f.relative_to(R)),'name':name,'type':typ})
assert len(inventory)==39,len(inventory)
# New native uses an identical observer/controller body and original GPU TUs.
assert (S/'observed_fit.inc').read_bytes()==(P/'observed_fit.inc').read_bytes()
for tag in ['cuda.jl','KA','JACC']:assert (S/f'tsne-julia-{tag}/main.jl').read_bytes()==(P/f'tsne-julia-{tag}/main.jl').read_bytes()
rows=[];total_alloc=total_bytes=0;sidecar_paths=[]
for m,v in d['models'].items():
 assert v['complete']and v['provenance_valid']and len(v['trials'])==5
 times=[x['time_seconds']for x in v['trials']];assert math.isclose(v['mean_seconds'],sum(times[1:])/4,rel_tol=1e-15)
 for trial in v['trials']:
  assert trial['correct']and trial['allocation_guard_pass'];f=Path(trial['allocation_guards']);g=json.load(open(f));assert g['pass'];allocs=g['allocations'];assert len(allocs)==(39 if m=='cuda'else 38);names=[];bytecount=0
  for item in allocs:
   assert item['pass']and item['guard_elements_each_end']==64;assert item['payload_bytes']>0;elem=item['element_bytes'];assert elem in[4,8];head=item['head_bytes'];tail=item['tail_bytes'];assert len(head)==len(tail)==64*elem
   if m=='cuda':expected=[165]*(64*elem);assert item['expected_byte']==165
   else:expected=item['expected_bytes'];assert len(expected)==64*elem;names.append(item['name'])
   assert head==expected and tail==expected;assert all(type(x)is int and 0<=x<=255 for x in head+tail);bytecount+=len(head)+len(tail)
  if m!='cuda':assert len(set(names))==38;assert set(names)=={'rawidx','dist','idx','prob','sym','beta','lower','upper','rowsum','entropy','found','points','attr','rep','gains','old','ones','norm','workspace','boxidx','xin','yin','tilde','xinterp','yinterp','potential','w','charges','boxlo','boxhi','kernel','fkernel','scratch','fin','fw','fout','spacing','den'}
  # Counterfactual CPU checker sensitivity only; no injected GPU failure claim.
  changed=list(allocs[0]['head_bytes']);changed[0]^=1;sensitivity_reject=changed!=(allocs[0].get('expected_bytes')or[165]*len(changed));assert sensitivity_reject
  row={'model':m,'trial':trial['trial'],'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'allocation_count':len(allocs),'actual_guard_bytes_checked':bytecount,'CPU_counterfactual_one_byte_change_rejected':True};rows.append(row);total_alloc+=len(allocs);total_bytes+=bytecount;sidecar_paths.append(str(f))
out={'status':'pass','result':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'configuration_id':d['configuration_id'],'guard_records':rows,'total_source_allocations_checked':total_alloc,'total_actual_head_tail_bytes_checked':total_bytes,'native_source_allocation_sites':inventory,'native_source_allocation_sites_count':39,'Julia_source_array_count':38,'new_nested_entries_byte_identical_to_old_entries':True,'native_observed_controller_text_byte_identical':True,'scientific_math_sources_changed':False,'coverage':'All explicit source GPU vectors/JuliaNamedTuple arrays in executed I1. Nativehelper7vectors plus32controller allocations, Juliaexistinghelperlayout38sourcearrays; differingcountsnotmissingguards. CUDA/Thrust/KA/JACC internal temporary pools, CPUmemory and unexecutedpaths notcovered. Canariescheckmodifiedhead/tailwrites, notallout-of-boundsreads/formalmemorysafety.','original_scientific_gaps_preserved':'I3exactgriddingfailures/earlierpointwiseFourierrejections and I1000no fulltrajectorycertificate remainunchanged. SamefixedoldI1revisedmetric; nofullscientificqualitypromotion.','timing':'Newguardstorage/init included; guardread/checktime subtracted alongwithoriginalobservers, neededcompletion/freeincluded. Fresh5all4 selectedbeforeexecution; alloldseries retained separately.','GPU_execution_by_this_audit':False};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in['status','total_source_allocations_checked','total_actual_head_tail_bytes_checked','native_source_allocation_sites_count','Julia_source_array_count']}))
