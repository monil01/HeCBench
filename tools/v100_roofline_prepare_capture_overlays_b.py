#!/usr/bin/env python3
"""Source-bound hidden capture-root overlays, group B (18 model slots).

Default emits immutable CPU-only plans/patch text. --stage only copies assets and
adapter text into fresh /tmp; no compiler, GPU, MPI, Julia or child is executed.
Output isolation does not establish replay safety or complete timed ROI coverage.
"""
import argparse, copy, difflib, hashlib, json, os, pathlib, re, shutil
P = pathlib.Path
SLOTS = {'sheath': ('cuda','cuda_jl','ka','jacc'),
         'snicit': ('cuda','cuda_jl','ka','jacc'), 'testSNAP': ('cuda',),
         'tpacf': ('cuda','cuda_jl','ka','jacc'),
         'zmddft': ('cuda','cuda_jl','ka','jacc'), 'sparkler': ('cuda',)}
def sha(p):
    h=hashlib.sha256()
    with P(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()
def dump(p,d): P(p).write_text(json.dumps(d,indent=2)+'\n')
def within(p,r): return P(p).resolve().is_relative_to(P(r).resolve())
def check(bindings):
    for p,h in bindings.items():
        if not P(p).is_file() or sha(p)!=h: raise ValueError('Input drift: '+p)
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1])
    a.add_argument('--base-plan-root',type=P,required=True)
    a.add_argument('--review-table',type=P,required=True)
    a.add_argument('--output-root',type=P,required=True)
    a.add_argument('--benchmark',action='append',choices=sorted(SLOTS))
    a.add_argument('--models',nargs='+',choices=['cuda','cuda_jl','ka','jacc'],default=['cuda','cuda_jl','ka','jacc'])
    a.add_argument('--stage',action='store_true')
    a.add_argument('--asset-cap-mib',type=int,default=512)
    a.add_argument('--reserve-mib',type=int,default=2048)
    z=a.parse_args();root=z.source_root.resolve();out=z.output_root.absolute();script=P(__file__).resolve()
    if min(z.asset_cap_mib,z.reserve_mib)<0: a.error('Nonnegative caps required')
    if not within(out,'/tmp') or out.resolve()==P('/tmp') or out.exists() or out.is_symlink(): a.error('Fresh output root strictly inside /tmp required')
    table=json.loads(z.review_table.read_text());reviews={(r['benchmark'],r['model']):r for r in table['models']}
    out.mkdir(parents=True);summary=[]
    for b in z.benchmark or sorted(SLOTS):
      for m in z.models:
        if m not in SLOTS[b]: continue
        bp=z.base_plan_root/b/m/'plan.json';base=json.loads(bp.read_text());review=reviews[b,m]
        if review['source_plan_sha256']!=sha(bp): raise ValueError('Reviewed base plan drift')
        if review['reference_result_sha256']!=base['reference_result_sha256'] or review['reference_config_sha256']!=base['raw_config_sha256']: raise ValueError('Review reference binding mismatch')
        if sha(review['source'])!=review['source_sha256']: raise ValueError('Reviewed source drift')
        result=P(base['reference_result'])
        if sha(result)!=base['reference_result_sha256'] or sha(result.with_name('config.json'))!=base['raw_config_sha256']: raise ValueError('Immutable reference/config drift')
        bindings=dict(base['execution_bindings']);check(bindings)
        additional={str(script):sha(script),str(bp.resolve()):sha(bp),str(z.review_table.resolve()):sha(z.review_table)}
        S=root/'src'/(b+'-julia');slot=out/b/m;slot.mkdir(parents=True);scratch=slot/'scratch';newsrc=scratch/'profile_source';private=scratch/'profile_outputs'/'private_capture';old=base['scratch_cwd']
        plan=copy.deepcopy(base)
        def rebase(v):
            if isinstance(v,str): return v.replace(old,str(scratch))
            if isinstance(v,list): return [rebase(x) for x in v]
            if isinstance(v,dict): return {k:rebase(x) for k,x in v.items()}
            return v
        for key in ('scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets'): plan[key]=rebase(base[key])
        patches=[];readonly={'canonical_source':str(S)}
        def read_bound(p):
            p=P(p).resolve();additional[str(p)]=sha(p);return p.read_text()
        def patch(src,dest,pairs,cpp=False):
            original=read_bound(src);text=original
            for before,after in pairs:
                if text.count(before)!=1: raise ValueError('Exact patch occurrence mismatch: '+str(src)+' '+before)
                text=text.replace(before,after,1)
            if cpp:
                def include(match):
                    target=(P(src).parent/match.group(1)).resolve()
                    read_bound(target)
                    return '#include '+json.dumps(str(target))
                text=re.sub(r'^#include "([^"]+)"',include,text,flags=re.M)
            if original==text: raise ValueError('Empty overlay')
            patches.append({'source':str(src),'source_sha256':sha(src),'copy':str(dest),'copy_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'diff':''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(src),tofile=str(dest)))})
        cmd=plan['profile_target_command'];build=plan['build_command']
        if b=='snicit':
            src=S/'study_run_model.py';dest=newsrc/src.name
            patch(src,dest,[('p=Path(__file__).resolve().parent','p=Path('+repr(str(S))+')'),("dest=p.parents[1]/'.porting-state/portable_julia/diagnostics/snicit'/d.name","dest=d/'retained_diagnostics'")])
            if cmd[1]!=str(src):raise ValueError('Unexpected wrapper position')
            cmd[1]=str(dest);read_bound(S/'audit_model.py')
        elif m=='cuda':
            src=S/'study_native.cu';dest=newsrc/src.name
            if b=='sheath':pairs=[('std::filesystem::path(__FILE__).parent_path()/"../../.porting-state/portable_julia/sheath_primary/cuda"','std::filesystem::path('+json.dumps(str(private/'cuda'))+')')]
            elif b=='testSNAP':pairs=[('auto source=std::filesystem::path(__FILE__).parent_path()','auto source=std::filesystem::path('+json.dumps(str(S))+')'),('auto captures=source/"../../.porting-state/portable_julia/testSNAP_primary"','auto captures=std::filesystem::path('+json.dumps(str(private))+')')]
            elif b=='tpacf':pairs=[('auto root=std::filesystem::path(__FILE__).parent_path()','auto root=std::filesystem::path('+json.dumps(str(S))+')'),('auto captures=root/"../../.porting-state/portable_julia/tpacf_primary"','auto captures=std::filesystem::path('+json.dumps(str(private))+')')]
            elif b=='zmddft':pairs=[('BASE=std::filesystem::path(__FILE__).parent_path()','BASE=std::filesystem::path('+json.dumps(str(S))+')'),('auto diagdir=BASE/"../../.porting-state/portable_julia/diagnostics/zmddft"','auto diagdir=std::filesystem::path('+json.dumps(str(private/'diagnostics'))+')')]
            elif b=='sparkler':pairs=[('std::filesystem::path(__FILE__).parent_path()/"../../.porting-state/portable_julia/sparkler_primary"','std::filesystem::path('+json.dumps(str(private))+')')]
            else:raise ValueError('Unsupported native case')
            patch(src,dest,pairs,cpp=True)
            if build.count(str(src))!=1:raise ValueError('Expected one native adapter compiler argument')
            build[build.index(str(src))]=str(dest)
        else:
            common=S/'study_common.jl';dest=newsrc/common.name
            if b=='sheath':pairs=[('const ROOT=normpath(@__DIR__)','const ROOT='+json.dumps(str(S))),('rawdir=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/sheath_primary",MODEL))','rawdir=joinpath(ENV["STUDY_OUTPUT_DIR"],"private_capture",MODEL)')]
            elif b=='tpacf':
                read_bound(S/'study_cases.jl')
                pairs=[('include(joinpath(@__DIR__,"study_cases.jl"))','include('+json.dumps(str(S/'study_cases.jl'))+')'),('const ROOT=normpath(@__DIR__)','const ROOT='+json.dumps(str(S))),('captures=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/tpacf_primary"))','captures=joinpath(ENV["STUDY_OUTPUT_DIR"],"private_capture")')]
            elif b=='zmddft':pairs=[('const BASE=@__DIR__','const BASE='+json.dumps(str(S))),('diagdir=joinpath(BASE,"../../.porting-state/portable_julia/diagnostics/zmddft")','diagdir=joinpath(ENV["STUDY_OUTPUT_DIR"],"private_capture","diagnostics")')]
            else:raise ValueError('Unsupported Julia case')
            patch(common,dest,pairs)
            entry=P(next(x for x in cmd if x.endswith('/main.jl')));newentry=newsrc/'main.jl';entrypairs=[]
            if b in ('sheath','zmddft'):entrypairs.append(('include("../study_common.jl")','include('+json.dumps(str(dest))+')'))
            else:entrypairs.append(('include(joinpath(@__DIR__,"..","study_common.jl"))','include('+json.dumps(str(dest))+')'))
            if b in ('sheath','tpacf'):
                read_bound(S/'study_device.jl');before='include("../study_device.jl")' if b=='sheath' else 'include(joinpath(@__DIR__,"..","study_device.jl"))'
                entrypairs.append((before,'include('+json.dumps(str(S/'study_device.jl'))+')'))
            patch(entry,newentry,entrypairs);cmd[cmd.index(str(entry))]=str(newentry)
        remaining=list(base['blocking_gates'])
        if base['runtime_asset_bytes']>z.asset_cap_mib*1024**2:remaining.append('capture_overlay_asset_copy_budget_exceeded')
        if build:
            if '-o' not in build or not within(build[build.index('-o')+1],scratch):raise ValueError('Compiler output not isolated')
        plan['execution_bindings']={**bindings,**additional};plan['blocking_gates']=remaining;plan['staged']=False
        plan['staging_status']='blocked' if remaining else 'input_plan_ready_ROI_and_kernel_mapping_still_required'
        plan['review']={'capture_output_root_overlay_group':'B','profile_only':True,'kernel_body_changes':False,'source_math_changes':False,'iteration_ROI_MPI_CPU_control_unchanged':True,'original_accepted_config_immutable':True,'cleared_exact_gates':[],'canonical_readonly_path_roles':readonly,'reason':['Redirect only historical capture roots; canonical input/header/device/helper paths retained.','No guard bypass or gate removal. Existing selected workload, checks, MPI ranks, stream/control and observer calls retained.','Fresh source adapter builds still required; CPU plan emission is not compilation or replay/ROI approval.']}
        plan['overlay_provenance']={'planner_sha256':sha(script),'base_plan_path':str(bp),'base_plan_sha256':sha(bp),'review_table_path':str(z.review_table),'review_table_sha256':sha(z.review_table),'source_result_sha256':base['reference_result_sha256'],'source_config_sha256':base['raw_config_sha256'],'profile_source_patches':patches,'additional_current_readonly_bindings':additional,'source_after':None,'compiler_commands_executed_by_planner':False,'GPU_target_executed':False}
        plan['notes']+=plan['review']['reason']
        if z.stage and not remaining:
            if shutil.disk_usage(out).free<base['runtime_asset_bytes']+z.reserve_mib*1024**2:raise ValueError('Free-space reserve prevents staging')
            scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir()
            for asset in plan['runtime_assets']:
                src=P(asset['source']);dst=P(asset['destination'])
                if not within(dst,scratch):raise ValueError('Runtime asset escapes scratch')
                for f in asset['source_files']:
                    if sha(f['path'])!=f['sha256']:raise ValueError('Runtime asset drift')
                dst.parent.mkdir(parents=True,exist_ok=True)
                if src.is_dir():shutil.copytree(src,dst)
                else:shutil.copy2(src,dst)
                for f in ([dst] if dst.is_file() else [x for x in dst.rglob('*') if x.is_file()]):f.chmod(0o555 if os.access(f,os.X_OK) else 0o444)
            for p in patches:
                dest=P(p['copy']);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(p['text']);dest.chmod(0o444)
                if sha(dest)!=p['copy_sha256']:raise ValueError('Overlay copy mismatch')
                plan['execution_bindings'][str(dest)]=p['copy_sha256']
            check(bindings);check(additional);plan['overlay_provenance']['source_after']=dict(additional);plan['staged']=True
        check(bindings);check(additional)
        dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in plan.items()if base.get(k)!=v})
        dump(slot/'plan.json',plan)
        summary.append({'benchmark':b,'model':m,'blocking_gates':remaining,'staged':plan['staged'],'plan':str(slot/'plan.json'),'plan_sha256':sha(slot/'plan.json'),'overlay_count':len(patches)})
    dump(out/'summary.json',{'models':summary,'GPU_executions':0,'compiler_executions':0,'shared_cache_mutations':False,'historical_captures_modified':False,'planner_sha256':sha(script),'qualification':'Hidden capture output isolation only; not executed-kernel/replay/fullROI approval.'})
    print(json.dumps({'models':len(summary),'ready':sum(not r['blocking_gates']for r in summary),'summary':str(out/'summary.json')}))
if __name__=='__main__':main()
