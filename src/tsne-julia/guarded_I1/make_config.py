from pathlib import Path
import json,argparse
S=Path(__file__).resolve().parent;P=S.parent;R=P.parents[1]
p=argparse.ArgumentParser();p.add_argument('--models',default='cuda');p.add_argument('--reuse-result',type=Path);p.add_argument('--native',type=Path,default=Path('/tmp/tsne-guarded-I1-native'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.load(open(P/'study_config_jacc_mnist_i1_metric_v4.json'));c={k:v for k,v in c.items()if not k.startswith('reuse_')};c['configuration_id']='tsne_mnist60000_K32_I1_revised_fourier_units_explicit_source_buffer_redzones_v5';c['input_contract']['guard_scope']='New head/tail64typedword guards for all explicit source native Thrustdevice_vectors and allJulia sourcearrays; complete actualguardbyte records afteruse/beforefree. Framework/Thrust internaltemporarypools and CPUmemory notcovered; canaries do not detect everyOOBread. No I3/I1000trajectory promotion.';c['guard_configuration']=True;c['preserved_old_configuration']='tsne_original_mnist60000_K32_boxes130_I1_lossless_fourier_units_revised_v4';c['storage_budget']={'persistent_captures_bytes':2605107760,'transient_peak_budget_bytes':4*1024**3,'storage':'own/tmpSTATE, no oldcapture deletion orrelocation'};c['models']={m:v for m,v in c['models'].items()if m in a.models.split(',')}
for m,spec in c['models'].items():
 if m=='cuda':spec['command'][-3]=str(a.native.resolve())
 else:
  cwd=S/('tsne-julia-'+{'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}[m]);spec['cwd']=str(cwd);spec['command']=[str(cwd/'main.jl')if arg==str(P/('tsne-julia-'+{'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}[m])/'main.jl')else arg for arg in spec['command']]
 extra=[S/'guard_plan.json']+[S/x for x in(['guard_allocator.h','study_native.cu','observed_fit.inc','build_native.py']if m=='cuda'else['study_common.jl','guarded_arrays.jl'])]
 if m=='cuda':extra+=[a.native.resolve()]
 else:extra+=[Path(spec['cwd'])/'main.jl']
 spec['source_dependencies']+=list(map(str,extra));spec['bind_all_declared_dependencies']=True;spec['native_check_description']=spec.get('native_check_description','')+' Newexplicit sourcearray head/tailredzones checked fromactualreadbacks withcompleteperallocation evidence; oldscientific scope unchanged.'
if a.reuse_result:
 old=json.load(open(a.reuse_result.resolve().with_name('config.json')));c['reuse_result']=str(a.reuse_result.resolve());c['reuse_models']=[m for m in c['models']if m!=a.models.split(',')[-1]]
 for m in c['reuse_models']:c['models'][m]=old['models'][m]
assert a.output.resolve().is_relative_to(S);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(c,indent=2)+'\n');print(a.output)
