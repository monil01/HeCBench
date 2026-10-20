#!/usr/bin/env python3
"""Prepare explicit case-specific roofline isolation overlays; never execute GPU targets.

Consumes immutable plans and a hash-pinned proposal file. Default plans only;
--stage copies inputs into a new /tmp tree. --build-manifest is a compiler-only
entry point intended for the collector's build_command, never called by staging.
No accepted source/config/result is modified. ROI/replay qualification is separate.
"""
import argparse, copy, difflib, hashlib, json, os, pathlib, shutil, subprocess, sys
sys.dont_write_bytecode = True
P = pathlib.Path
CASES = {'xlqc', 'interval', 'pad', 'leukocyte', 'ans', 'amgmk'}
SUFFIX = {'.jl','.py','.cu','.cuh','.c','.cc','.cpp','.h','.hpp','.inc','.toml'}

def sha(p):
    h = hashlib.sha256()
    with P(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def under(p, root):
    try: P(p).resolve().relative_to(P(root).resolve()); return True
    except ValueError: return False

def dump(p, obj):
    P(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def check(bound):
    for p, h in bound.items():
        if not P(p).is_file() or sha(p) != h: raise ValueError('Binding drift: '+p)

def manifest_tree(root):
    if root.is_symlink(): raise ValueError('Tree root symlink requires separate review: '+str(root))
    records=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError('Tree symlink requires separate review: '+str(p))
        if p.is_file(): records.append({'relative_path':str(p.relative_to(root)), 'sha256':sha(p), 'bytes':p.stat().st_size})
    return sorted(records,key=lambda x:x['relative_path'])

def build_only(path, expected):
    if sha(path)!=expected: raise ValueError('Build manifest hash mismatch')
    d=json.loads(path.read_text()); scratch=P(d['scratch']).resolve()
    if not under(scratch, '/tmp') or scratch==P('/tmp') or P.cwd().resolve()!=scratch: raise ValueError('Build cwd must be isolated scratch')
    if not d.get('compiler_only'): raise ValueError('Not a compiler-only manifest')
    check(d['input_bindings']); before=dict(d['input_bindings'])
    receipt=scratch/'overlay_build_outcome.json'
    if receipt.exists(): raise ValueError('Refuse repeated build into used scratch')
    for p in d['must_be_absent_before_build']:
        if P(p).exists() or P(p).is_symlink(): raise ValueError('Existing build output: '+p)
    result={'compiler_commands':[], 'GPU_target_executed':False, 'source_before':before}
    try:
        for command in d['commands']:
            if P(command[0]).name not in {'g++','nvcc','make'}: raise ValueError('Compiler command outside explicit allowlist')
            result['compiler_commands'].append(command)
            subprocess.run(command, cwd=scratch, check=True)
        for item in d.get('generated_text',[]):
            dest=P(item['path'])
            if not under(dest,scratch) or dest.exists(): raise ValueError('Unsafe generated pointer')
            dest.write_text(item['text'])
        result['outputs']=[{'path':p,'sha256':sha(p)}for p in d['final_outputs']]
        result['status']='compiler_commands_completed_no_target_execution'
    finally:
        check(before); result['source_after']=dict(before)
        result['source_bytes_unchanged']=True; dump(receipt,result)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1])
    ap.add_argument('--base-plan-root',type=P); ap.add_argument('--proposals',type=P);ap.add_argument('--proposals-sha256')
    ap.add_argument('--output-root',type=P);ap.add_argument('--benchmark',action='append',choices=sorted(CASES))
    ap.add_argument('--models',nargs='+',choices=['cuda','cuda_jl','ka','jacc'],default=['cuda','cuda_jl','ka','jacc'])
    ap.add_argument('--stage',action='store_true');ap.add_argument('--asset-cap-mib',type=int,default=512);ap.add_argument('--reserve-mib',type=int,default=2048)
    ap.add_argument('--build-manifest',type=P);ap.add_argument('--manifest-sha256')
    a=ap.parse_args()
    if a.build_manifest:
        if not a.manifest_sha256: ap.error('Build manifest needs exact SHA256')
        build_only(a.build_manifest,a.manifest_sha256);return
    if not all([a.base_plan_root,a.proposals,a.proposals_sha256,a.output_root]): ap.error('Base plans, proposals, exact proposal SHA256 and fresh output root required')
    if min(a.asset_cap_mib,a.reserve_mib)<0: ap.error('Nonnegative storage bounds required')
    root=a.source_root.resolve();out=a.output_root.absolute();script=P(__file__).resolve()
    if not under(out,'/tmp') or out.resolve()==P('/tmp') or out.exists() or out.is_symlink(): ap.error('Fresh output root strictly under /tmp required')
    if sha(a.proposals)!=a.proposals_sha256: raise ValueError('Proposal SHA mismatch')
    proposals=json.loads(a.proposals.read_text())['proposals']
    # These two output-only overlays bind the immutable model plan directly.
    for b in ('ans','amgmk'):
        if not a.benchmark or b not in a.benchmark: continue
        for m in a.models:
            f=a.base_plan_root/b/m/'plan.json'
            d=json.loads(f.read_text())
            proposals.append({'benchmark':b,'model':m,'source_result_sha256':d['reference_result_sha256'],'source_config_sha256':d['raw_config_sha256'],'execution_bindings':d['execution_bindings'],'actions':[],'disposition':'exact_profile_copy_path_only_overlay'})
    out.mkdir(parents=True);summary=[]
    for proposal in proposals:
        b=proposal['benchmark'];m=proposal['model']
        if b not in CASES or (a.benchmark and b not in a.benchmark) or m not in a.models:continue
        if b=='interval' and m!='cuda': raise ValueError('Only native interval overlay is reviewed')
        basepath=a.base_plan_root/b/m/'plan.json';base=json.loads(basepath.read_text());plan=copy.deepcopy(base)
        if (base['reference_result_sha256']!=proposal['source_result_sha256'] or base['raw_config_sha256']!=proposal['source_config_sha256'] or base['execution_bindings']!=proposal['execution_bindings']): raise ValueError('Proposal/base join mismatch '+b+'/'+m)
        rp=P(base['reference_result']);cp=rp.with_name('config.json')
        if sha(rp)!=base['reference_result_sha256'] or sha(cp)!=base['raw_config_sha256']:raise ValueError('Original evidence changed')
        check(base['execution_bindings']);check({x['path']:x['sha256']for x in proposal.get('source_evidence',[])});slot=out/b/m;scratch=slot/'scratch';old=base['scratch_cwd'];slot.mkdir(parents=True)
        def move(x):return x.replace(old,str(scratch)) if isinstance(x,str) else x
        for key in ['build_command','profile_target_command']:plan[key]=[move(x)for x in plan[key]]
        plan['scratch_cwd']=str(scratch);plan['environment_overrides']={k:move(v)for k,v in plan['environment_overrides'].items()}
        for asset in plan['runtime_assets']:asset['destination']=move(asset['destination'])
        bindings=dict(base['execution_bindings']);bindings[str(script)]=sha(script);bindings[str(a.proposals.resolve())]=a.proposals_sha256;bindings[str(cp)]=sha(cp);bindings[str(basepath.resolve())]=sha(basepath)
        evidence={}
        for directory in [root/'src'/(b+'-julia'),root/'src'/(b+'-cuda')]:
            for p in directory.rglob('*'):
                if p.is_file() and p.suffix in SUFFIX:evidence[str(p.resolve())]=sha(p)
        copies=[];precreate=[];clear=[];notes=[];commands=[];texts=[];outputs=[];absent=[];patches=[]
        def patched(source,destination,changes):
            original=source.read_text();revised=original
            expected=base['execution_bindings'].get(str(source.resolve()))
            if not expected or sha(source)!=expected:raise ValueError('Patched source not bound by accepted model')
            for oldtext,newtext in changes:
                if revised.count(oldtext)!=1:raise ValueError('Patch requires unique exact source fragment: '+str(source))
                revised=revised.replace(oldtext,newtext)
            record={'original':str(source.resolve()),'original_sha256':sha(source),'copy':str(destination),'copy_sha256':hashlib.sha256(revised.encode()).hexdigest(),'text':revised,'changes':[{'before':x,'after':y}for x,y in changes],'unified_diff':list(difflib.unified_diff(original.splitlines(True),revised.splitlines(True),fromfile=str(source),tofile=str(destination)))}
            patches.append(record);return destination
        actions=proposal['actions']
        for action in actions:
            if action['kind']=='exact_argv_replacement':
                i=action['index']
                if plan['profile_target_command'][i]!=action['original']:raise ValueError('Exact argv mismatch')
                dest=action['replacement'].replace('{scratch}',str(scratch))
                if not under(dest,scratch):raise ValueError('Arg destination escape')
                plan['profile_target_command'][i]=dest
            elif action['kind']=='copy_input_tree_to_fresh_scratch':
                src=P(action['source']);actual=manifest_tree(src)
                expected=sorted(action['files'],key=lambda x:x['relative_path'])
                if actual!=expected:raise ValueError('Fixture tree changed')
                dest=P(action['destination'].replace('{scratch}',str(scratch)))
                copies.append({'source':str(src),'destination':str(dest),'files':actual,'writable_derived_paths':action['writable_derived_paths']})
                evidence.update({str(src/x['relative_path']):x['sha256']for x in actual})
            elif action['kind']=='create_new_scratch_directory':precreate.append(action['destination'].replace('{scratch}',str(scratch)))
        if b=='ans':
            source=root/'src/ans-julia/study_run_model.py';destination=scratch/'profile_source/study_run_model.py'
            old="P=Path(__file__).resolve().parent;D=Path(os.environ['STUDY_OUTPUT_DIR']);dest=P.parents[1]/'.porting-state/portable_julia/diagnostics/ans'/D.name;dest.mkdir(parents=True,exist_ok=True)"
            new="P=Path("+repr(str(source.parent))+");D=Path(os.environ['STUDY_OUTPUT_DIR']);dest=D/'diagnostics';dest.mkdir(parents=True,exist_ok=True)"
            patched(source,destination,[(old,new)])
            if plan['profile_target_command'][1]!=str(source):raise ValueError('Unexpected ANS wrapper position')
            plan['profile_target_command'][1]=str(destination)
            notes.append('Only wrapper source lookup and diagnostic destination changed; P remains canonical source for audit_model.py. sys.argv[1:] child command unchanged.')
        elif b=='amgmk':
            S=root/'src/amgmk-julia';raw=scratch/'profile_outputs/raw';precreate.append(str(raw))
            if m=='cuda':
                source=S/'study_native.cu';dest=scratch/'profile_source/study_native.cu'
                changes=[('#include "../amgmk-cuda/main.cu"','#include '+json.dumps(str(root/'src/amgmk-cuda/main.cu'))),('#include "study_params.h"','#include '+json.dumps(str(S/'study_params.h'))),('auto rawdir=root/".porting-state/porting_logs/amgmk-julia-shared/raw/cuda";','auto rawdir=std::filesystem::path('+json.dumps(str(raw))+');')]
                patched(source,dest,changes)
                if plan['build_command'].count(str(source))!=1:raise ValueError('Unexpected AMGmk native source argv')
                plan['build_command']=[str(dest)if x==str(source)else x for x in plan['build_command']]
            else:
                common=S/'study_common.jl';dest=scratch/'profile_source/study_common.jl'
                old='rawdir=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","amgmk-julia-shared","raw",model);mkpath(rawdir)'
                revised=common.read_text().replace(old,'rawdir='+json.dumps(str(raw))+';mkpath(rawdir)')
                if common.read_text().count(old)!=1:raise ValueError('AMGmk rawdir fragment drift')
                # Preserve all other canonical include/fixture roots explicitly.
                changes=[(old,'rawdir='+json.dumps(str(raw))+';mkpath(rawdir)')]
                for line in revised.splitlines():
                    if '@__DIR__' in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(S)))))
                patched(common,dest,changes)
                entry=next(P(x)for x in plan['profile_target_command']if x.endswith('/main.jl'))
                newentry=scratch/'profile_source/main.jl';text=entry.read_text();changes=[]
                oldinclude='include(joinpath(@__DIR__,"..","study_common.jl"))'
                changes.append((oldinclude,'include('+json.dumps(str(dest))+')'))
                for line in text.replace(oldinclude,'').splitlines():
                    if '@__DIR__'in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(entry.parent)))))
                patched(entry,newentry,changes)
                plan['profile_target_command']=[str(newentry)if x==str(entry)else x for x in plan['profile_target_command']]
            notes.append('No AMGMK_RAW_DIR getenv exists: output root is source-derived. Copies change raw storage path plus canonical include/fixture lookup only; all kernel math/host loops/R500 unchanged.')
        elif b=='xlqc':
            # Original builder is one nvcc subprocess with explicit -o, no cache/make.
            clear=['custom_build_script_requires_explicit_output_and_side_effect_review','source_bound_absolute_path_literals_require_write_isolation_review']
            notes.append('Fixture clone allows only derived basis files writable; original fixture bytes remain unchanged. prepare=0 remains unchanged.')
        elif b=='pad':
            S=root/'src/pad-julia';N=root/'src/pad-cuda';lib=scratch/'profile_overlay_build/libpadcpu.so'
            commands=[['g++','-O3','-std=c++17','-fPIC','-shared','-pthread',str(S/'cpu_bridge.cpp'),str(N/'host_task.cpp'),'-o',str(lib)]]
            texts=[{'path':str(scratch/'cpu_component_path.txt'),'text':str(lib)+'\n'}];outputs=[str(lib),texts[0]['path']];absent=list(outputs)
            if m=='cuda':
                exe=scratch/'study_native';commands.append(['nvcc','-O3','-std=c++17','-arch=sm_70',str(S/'study_cuda.cu'),str(N/'kernel.cu'),'-L'+str(lib.parent),'-lpadcpu','-Xlinker','-rpath','-Xlinker',str(lib.parent),'-o',str(exe)]);outputs.append(str(exe));absent.append(str(exe));plan['profile_target_command'][0]=str(exe)
            clear=['custom_build_script_requires_explicit_output_and_side_effect_review','source_bound_absolute_path_literals_require_write_isolation_review']
            notes.append('Old shared-cache builder not invoked. CPU/GPU flag cooperation needs separate replay qualification.')
        elif b=='leukocyte':
            if m=='cuda':
                S=root/'src/leukocyte-julia';src=S/'corrected_native/meschach_lib';dest=scratch/'profile_overlay_build/meschach_lib'
                files=[x for x in manifest_tree(src)if P(x['relative_path']).suffix not in {'.o','.a','.so'}]
                copies.append({'source':str(src),'destination':str(dest),'files':files,'writable_derived_paths':[]})
                evidence.update({str(src/x['relative_path']):x['sha256']for x in files})
                commands=[['make','-C',str(dest),'all','CC=g++','CFLAGS=-O3 -std=c++17 -fPIC -x c++']];objects=[]
                sources=[S/'study_native_driver.cu']+[S/'instrumented_native'/n for n in ['detect_main.c','track_ellipse.c','track_ellipse_gpu.c','find_ellipse.c','misc_math.c','avilib.c']]
                for i,p in enumerate(sources):
                    obj=scratch/'profile_overlay_build'/('native_'+str(i)+'.o');objects.append(str(obj));commands.append(['/usr/bin/nvcc','-O3','-std=c++17','-arch=sm_70','-I'+str(dest),'-x','cu','-c',str(p),'-o',str(obj)])
                exe=scratch/'study_native';commands.append(['/usr/bin/nvcc',*objects,str(dest/'meschach.a'),'-lm','-o',str(exe)]);outputs=[str(dest/'meschach.a'),str(exe)];absent=outputs+objects;plan['profile_target_command'][0]=str(exe)
                clear=['custom_build_script_requires_explicit_output_and_side_effect_review']
            notes.append('Capture root passed via exact argv; native begin_case uses that root; Julia raw/guard/layout/timing and rejection files use ARGS[2]. Legacy result.txt remains cwd-relative scratch.')
        # Clear only the exact output paths that were actually replaced.
        for action in actions:
            if action['kind']=='exact_argv_replacement':
                gate='absolute_directory_argument_requires_role:'+action['original']
                if gate in base['blocking_gates']:clear.append(gate)
        remaining=[g for g in base['blocking_gates']if g not in clear]
        if any(g.startswith(('original_contract_requires_multiple_devices','effective_timing_region_missing','runtime_asset_destination_escape','missing_runtime_asset','runtime_asset_copy_budget','build_output_outside'))for g in clear):raise ValueError('Forbidden gate removal')
        allbytes=plan['runtime_asset_bytes']+sum(sum(x['bytes']for x in c['files'])for c in copies)
        plan['runtime_asset_bytes']=allbytes
        if allbytes>a.asset_cap_mib*1024**2:remaining.append('overlay_copy_budget_exceeded')
        if commands:
            buildspec={'compiler_only':True,'scratch':str(scratch),'commands':commands,'generated_text':texts,'final_outputs':outputs,'must_be_absent_before_build':absent,'input_bindings':{**bindings,**evidence}}
            bp=slot/'build_manifest.json';dump(bp,buildspec);bindings[str(bp)]=sha(bp)
            plan['build_command']=[sys.executable,str(script),'--build-manifest',str(bp),'--manifest-sha256',sha(bp)]
        plan['blocking_gates']=sorted(set(remaining));plan['staging_status']='blocked'if remaining else'input_plan_ready_ROI_and_kernel_mapping_still_required';plan['execution_bindings']=bindings
        review={'case_specific_overlay':True,'profile_only':True,'kernel_body_changes':False,'source_math_changes':False,'original_accepted_config_immutable':True,'cleared_exact_gates':[g for g in base['blocking_gates']if g in clear],'additional_current_readonly_input_bindings':evidence,'reason':notes,'source_result_sha256':proposal['source_result_sha256'],'source_config_sha256':proposal['source_config_sha256'],'proposal':proposal}
        plan['review']={'path':str(a.proposals.resolve()),'sha256':a.proposals_sha256,'entry':review};plan['notes']+=notes
        plan['overlay_provenance']={'planner_sha256':sha(script),'base_plan_path':str(basepath.resolve()),'base_plan_sha256':sha(basepath),'proposal_file_sha256':a.proposals_sha256,'source_before':evidence,'source_after':None,'copies':copies,'profile_source_patches':patches,'precreated_directories':precreate,'build_commands_executed_by_planner':False,'GPU_target_executed':False}
        if a.stage and not remaining:
            if shutil.disk_usage(out).free<allbytes+a.reserve_mib*1024**2:raise ValueError('Free space reserve prevents overlay staging')
            scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();(scratch/'profile_overlay_build').mkdir()
            for asset in plan['runtime_assets']:
                src=P(asset['source']);dest=P(asset['destination'])
                for item in asset['source_files']:
                    if sha(item['path'])!=item['sha256']:raise ValueError('Runtime asset changed')
                dest.parent.mkdir(parents=True,exist_ok=True)
                if src.is_dir():shutil.copytree(src,dest)
                else:shutil.copy2(src,dest)
                for f in [dest]if dest.is_file()else[x for x in dest.rglob('*')if x.is_file()]:f.chmod(0o555 if os.access(f,os.X_OK)else 0o444)
            for c in copies:
                dest=P(c['destination']);dest.mkdir(parents=True)
                for item in c['files']:
                    src=P(c['source'])/item['relative_path'];f=dest/item['relative_path'];f.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,f)
                    if sha(f)!=item['sha256']:raise ValueError('Copy mismatch')
                    f.chmod(0o644 if item['relative_path']in c['writable_derived_paths']else 0o444)
                    if item['relative_path']not in c['writable_derived_paths']:plan['execution_bindings'][str(f)]=item['sha256']
            for patch in patches:
                dest=P(patch['copy']);dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(patch['text']);dest.chmod(0o444)
                if sha(dest)!=patch['copy_sha256']:raise ValueError('Patched source write mismatch')
                plan['execution_bindings'][str(dest)]=patch['copy_sha256']
            for p in precreate:P(p).mkdir(parents=True)
            check(bindings);check(evidence);plan['overlay_provenance']['source_after']=dict(evidence);plan['overlay_provenance']['source_bytes_unchanged_after_staging']=True;plan['staged']=True
        else:plan['staged']=False
        diff={k:{'before':base.get(k),'after':plan.get(k)}for k in plan if base.get(k)!=plan[k]}
        dump(slot/'plan_diff.json',diff);dump(slot/'plan.json',plan);summary.append({'benchmark':b,'model':m,'staged':plan['staged'],'blocking_gates':plan['blocking_gates'],'copy_bytes':allbytes,'plan':str(slot/'plan.json')})
    dump(out/'summary.json',{'models':summary,'GPU_executions':0,'compiler_executions':0,'source_mutations':False,'ROI_qualification':'Unchanged: partial kernel inventory is not whole timed ROI.','planner_sha256':sha(script)})
    print(json.dumps({'plans':len(summary),'ready':sum(not x['blocking_gates']for x in summary),'staged':sum(x['staged']for x in summary),'summary':str(out/'summary.json')}))
if __name__=='__main__':main()
