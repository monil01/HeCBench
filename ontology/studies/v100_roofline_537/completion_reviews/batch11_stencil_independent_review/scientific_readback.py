import json,hashlib,gzip,fcntl
from pathlib import Path
O=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');P=O.parent/'roi_membership_batch11_stencil';sha=lambda b:hashlib.sha256(b).hexdigest();d=json.loads((P/'proposed_stage_patches.json').read_bytes());review=json.loads((O/'review.json').read_bytes());out=[]
def ev(rel,needles):
 p=W/rel;b=p.read_bytes();assert sha(b)==review['input_bindings'][str(p)]['sha256'];lines=b.decode().splitlines(keepends=True);spans=[]
 for needle in needles:
  ids=[i for i,l in enumerate(lines)if needle in l];assert ids,(rel,needle)
  for i in ids:
   st=sum(len(l.encode())for l in lines[:i]);en=st+len(lines[i].encode());assert b[st:en]==lines[i].encode();spans.append({'line':i+1,'start_byte':st,'end_byte':en,'text':lines[i]})
 return {'path':str(p),'sha256':sha(b),'spans':spans}
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 out=[
 {'case':'mixbench CUDAjl','finding':'Exact selected source wrapper calls benchmark_kernel!; approve only this callable value and its matching saved family. No generic selector broadening.','evidence':ev('src/mixbench-julia/mixbench-julia-cuda.jl/study.jl',['launch_julia!'])},
 {'case':'MMCSF','finding':'Secondary FP64 output allocation/initialization and per-mode D2H projection to Float32/H2D/free occur before Observe starts. Four full source cases and final completion minus excluded observation intervals form net host-inclusive timer; not original pure FP32 or ALS feedback.','evidence':ev('src/mmcsf-julia/secondary_fp64_atomic/full_study.cpp',['struct Observe','void mm_mode_begin','void mm_mode_end','for(trial_id=0;'])},
 {'case':'MNIST','finding':'Pause/paused synchronizes before accounting elapsed time, so completion belongs to timed interval; snapshots/readbacks inside paused body excluded. Actual bounded learn2/test workflow retains source host/data/memset work. Native repeated apply-step/apply-grad selectors are aliases, not removed source invocations.','evidence':[ev('src/mnist-julia/study_observer.h',['struct Pause']),ev('src/mnist-julia/study_common.jl',['if active;CUDA.device_synchronize()','s.mark=time_ns();s.running=true;learn!'])]},
 {'case':'MiniBUDE','finding':'One Work.run/run_bude warmup invocation per case precedes five trials. Each invocation has seven internal launches, timed as per-repeat mean; six case means summed. Draft300? wording is not observed warmup evidence and cannot enter paper as fact; MLP separately has actual300 source warmups.','evidence':[ev('src/minibude-julia/study_native.cu',['double run()','int main(){static_assert']),ev('src/minibude-julia/study_common.jl',['const REPEATS=7','for w in work;run_bude(w);end'])]},
 {'case':'outside diagnostics','finding':'Michalewicz full objective capture and MiniWeather mass/energy diagnostics lie outside timers; MIS init-priority replay occurs after timed final state is read. Same family may appear in both contexts, so sampled phase remains unproved.','evidence':[ev('src/michalewicz-julia/study_common.jl',['push!(times,(time_ns()-start)']),ev('src/miniWeather-julia/study_native.cu',['auto d0=diag(c)']),ev('src/mis-julia/study_common.jl',['final=Array(c.sr)','launch(1,c.n,c.e,c.id,c.ld,c.pd)'])]}
 ]
 aliases=[(r['record_key'],s)for r in d['patches']for s in r['stage_patches']if s['updates'].get('alias_of_stage_id')]
 assert len(aliases)==4 and all(k['benchmark']=='mnist'and k['model']=='cuda'and s['updates']['semantic_family_in_primary_ROI']and s['updates']['model_kernel_regex']is None for k,s in aliases)
 assert len(d['withheld_models'])==7 and len(d['withheld_stages'])==10
 for x in review['input_bindings'].items():assert sha(Path(x[0]).read_bytes())==x[1]['sha256']
 (O/'scientific_readback.json').write_text(json.dumps({'status':'source_interpretation_consistent_pending_root_adoption','findings':out,'four_MNIST_alias_nulls_verified':True,'seven_model_holds_and_ten_unresolved_stages_preserved':True,'sample_phase':'unproved','whole_ROI':False,'causal_claim':False},indent=2)+'\n')
print('Independent source focus, four aliases, seven holds, ten unchanged unresolved stages pass')
