"""Supplemental actual untimed five-stage target trace; never primary timing samples."""
from pathlib import Path
import json,hashlib,subprocess,os,struct,base64
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;D=R/'.porting-state/portable_julia/diagnostics/cc';D.mkdir(parents=True,exist_ok=True)
plan=json.loads((P/'study_plan.json').read_text());out=[]
for model,folder,project in [('cuda_jl','cc-julia-cuda.jl','_julia_ka_env'),('ka','cc-julia-KA','_julia_ka_env'),('jacc','cc-julia-JACC','_julia_jacc_env')]:
 source=P/folder/'main.jl';sha=hashlib.sha256(source.read_bytes()).hexdigest();prefix=source.read_text().rsplit('cc_study(',1)[0]
 setup='using Base64;Base.include_string(Main,String(base64decode('+json.dumps(base64.b64encode(prefix.encode()).decode())+')),'+json.dumps(str(source))+')\n'
 launch={'cuda_jl':'f=(init_kernel!,low_kernel!,medium_kernel!,high_kernel!,flatten_kernel!)[stage];CUDA.@cuda threads=256 blocks=c.blocks f(c.n,idx,adj,stat,wl,ctr)','ka':'STAGES[stage](c.n,idx,adj,stat,wl,ctr;ndrange=c.blocks*256)','jacc':'JACC.parallel_for(SPEC,c.blocks*256,STAGES[stage],c.n,idx,adj,stat,wl,ctr)'}[model]
 code=setup+'\nalloc='+('JACC.array'if model=='jacc'else'CuArray')+'\n'+r'''
for c in CC_CASES
 n=Int(c.n);a=alloc(cc_read("case$(c.id)_idx_i32.bin",c.ic));b=alloc(cc_read("case$(c.id)_adj_i32.bin",c.ac));s=alloc(fill(CC_GUARD,n+32));w=alloc(fill(CC_GUARD,n+32));ctr=alloc(zeros(Int32,4));idx=view(a,17:length(a)-16);adj=view(b,17:length(b)-16);stat=view(s,17:n+16);wl=view(w,17:n+16)
 for stage in 1:5
 LAUNCH
 CUDA.device_synchronize(blocking=true);labels=Array(stat);counters=Array(ctr)
 @printf("STUDY_TRACE {\"case\":%d,\"stage\":%d,\"topL\":%d,\"topH\":%d,\"labels\":[",c.id,stage-1,counters[1],counters[3]);for i in eachindex(labels);i>1&&print(',');print(labels[i]);end;println("]}");flush(stdout)
 end
end
'''.replace('LAUNCH',launch)
 log=D/f'{model}_stages.log';cmd=['flock','-x','/tmp/hecbench-julia-study-gpu.lock','env','CUDA_VISIBLE_DEVICES=6','julia','--project='+str(R/'src'/project),'-e',code]
 with log.open('w')as f:proc=subprocess.run(cmd,cwd=P,stdout=f,stderr=subprocess.STDOUT)
 assert proc.returncode==0,(model,log);assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
 rows=[json.loads(x.removeprefix('STUDY_TRACE '))for x in log.read_text().splitlines()if x.startswith('STUDY_TRACE ')];assert len(rows)==15
 checks=[]
 for row in rows:
  c=plan['cases'][row['case']];expected=struct.unpack('<'+'i'*c['n'],(P/f"case{c['id']}_labels_i32.bin").read_bytes());labels=row['labels'];assert len(labels)==c['n'];assert all(0<=p<=v and expected[p]==expected[v]for v,p in enumerate(labels))
  if row['stage']==4:assert list(expected)==labels
  checks.append(dict(case=c['id'],stage=row['stage'],full_values_checked=len(labels),same_component_monotone_parent=True,final_BFS_minimum_exact=row['stage']==4,topL=row['topL'],topH=row['topH']))
 out.append(dict(model=model,source=dict(path=str(source.relative_to(R)),sha256=sha),scope='One fresh supplemental untimed process,all5stages/all3shapes; not sameexecution asprimaryfivetrials, no extra primarytimes.',command=cmd,exit_status=proc.returncode,log=dict(path=str(log.relative_to(R)),sha256=hashlib.sha256(log.read_bytes()).hexdigest()),checks=checks));print('CC_TARGET15_STAGES_PASS',model,flush=True)
(P/'target_stage_diagnostics.json').write_text(json.dumps(dict(records=out,actual_raw_trace_not_for_git=True),indent=2)+'\n')
