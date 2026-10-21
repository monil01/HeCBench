from pathlib import Path
import hashlib,difflib,json
D=Path(__file__).resolve().parent;B=Path('/tmp/roofline-completion-20261021/roi_membership_root_merge_v4_preparation/merge_reviews_v4.py');old=B.read_text();v=old
expected='ee3068f93f97b2a618b2c5aafe08e39056b5eb234e8705d0a597f5545360b5ee'
if hashlib.sha256(B.read_bytes()).hexdigest()!=expected:raise ValueError('Exact frozen v4 base required')
def change(a,b):
 global v
 if v.count(a)!=1:raise ValueError('Ambiguous source-authoring anchor '+a[:80])
 v=v.replace(a,b,1)
change('def validate(root,packets,transaction):','def validate(root,packets,transaction,scientific_path):')
change(" scientific_path=Path('/tmp/roofline-completion-20261021/roi_v3_root_scientific_review.json');scientific_raw=file_bytes(scientific_path);scientific=json.loads(scientific_raw)"," scientific_raw=file_bytes(scientific_path);scientific=json.loads(scientific_raw)\n expected_count_keys={'model_records','names','inclusion_declarations','exclusions','pure_alias_nulls','inclusion_alias_nulls','non_null_inclusion_stages'}\n expected_counts=scientific.get('expected_counts')\n need(isinstance(expected_counts,dict)and set(expected_counts)==expected_count_keys,'Scientific review must specify exactly all seven expected-count fields')\n need(all(type(x)is int and x>=0 for x in expected_counts.values()),'Expected counts must be nonnegative integers')")
change(" need(counts['model_records']==56 and counts['names']==14 and counts['inclusion_declarations']==124,'Expected three-packet population not validated')"," need(counts==expected_counts,'Complete seven-field root-reviewed population not validated')")
change("ap.add_argument('--output-root',type=Path,required=True);", "ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--root-scientific-review',type=Path,required=True);")
change("validate(root,[x.resolve()for x in a.packets],token)","validate(root,[x.resolve()for x in a.packets],token,a.root_scientific_review.resolve())")
v=v.replace('roi-v4-','roi-v5-').replace('Fresh v4 transaction','Fresh v5 transaction').replace('Reviewed v4 plan','Reviewed v5 plan').replace('EXPLICIT v4 metadata','EXPLICIT v5 metadata')
change("   need(not marker.exists()and not marker.is_symlink(),'UNFINISHED TRANSACTION: root recovery review required; do not start another merge')", "   previous_marker=study/'.roi-v4-active-commit.json'\n   need(all(not p.exists()and not p.is_symlink()for p in (previous_marker,marker)),'UNFINISHED v4/v5 TRANSACTION: root recovery review required; do not start another merge')")
clock=r'''  def clock_overlay(e):
   need(b=='clock'and m in ('cuda_jl','ka','jacc'),'Clock overlay restricted to selected Julia3')
   patches=plan.get('capture_overlay_provenance',{}).get('patches',[])
   matches=[(i,p)for i,p in enumerate(patches)if p['copy']==e['actual_executed_source_path']]
   need(len(matches)==1,'Retained clock overlay text missing/duplicate');i,p=matches[0]
   need(e.get('selected_plan')==cap['plan']and binding(e['selected_plan'])==plan,'Clock retained-text selected-plan binding mismatch')
   need(e.get('text_JSON_pointer')=='/capture_overlay_provenance/patches/'+str(i)+'/text','Clock retained-text pointer mismatch')
   raw=p['text'].encode();need(e['retained_text']==p['text']and sha(raw)==p['copy_sha256']==e['actual_executed_sha256'],'Retained clock complete copy byte/SHA mismatch')
   copy_path=Path(p['copy']);original_path=Path(p['original']);parent=root/'src/clock-julia'
   need(copy_path.parent==Path(plan['scratch_cwd'])/'profile_source','Clock historical copied source outside exact selected scratch')
   need(p['copy_sha256']in captured[str(copy_path.resolve())]and p['original_sha256']in captured[str(original_path.resolve())],'Clock original/copy absent from actual captured bindings')
   need(p['original']==e['original_bound_path']and p['original_sha256']==e['original_sha256'],'Clock original identity join')
   original_raw=file_bytes(original_path);need(sha(original_raw)==p['original_sha256'],'Clock present original source drift');original=original_raw.decode();source_hashes[str(original_path.resolve())]=p['original_sha256']
   # Independently construct EXACT known path-only rules. Neither the plan's
   # changes array nor packet booleans can authorize arbitrary source changes.
   if original_path==parent/'study_common.jl':
    need(copy_path.name=='study_common.jl','Clock common copied filename')
    before='rawdir=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","clock-julia-"*model,"raw_timings")'
    after='rawdir=joinpath(ENV["STUDY_OUTPUT_DIR"],"private_clock")'
    fixture=' words=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(words)==1042'
    changes=[(before,after),(fixture,fixture.replace('@__DIR__',json.dumps(str(parent))))]
   else:
    relative={'cuda_jl':'clock-julia-cuda.jl/study.jl','ka':'clock-julia-KA/main.jl','jacc':'clock-julia-JACC/main.jl'}[m]
    need(original_path==parent/relative and copy_path.name=='main.jl','Clock model-specific entry identity')
    commons=[z for z in patches if z.get('original')==str(parent/'study_common.jl')and z.get('copy')==str(copy_path.parent/'study_common.jl')]
    need(len(commons)==1,'Clock entry same historical copied-common relationship missing/ambiguous');common=commons[0]
    need(sha(common['text'].encode())==common['copy_sha256']and common['copy_sha256']in captured[str(Path(common['copy']).resolve())],'Clock included common text/captured binding mismatch')
    # Validate the common independently even when an entry-only packet arrives.
    common_original=file_bytes(parent/'study_common.jl');need(sha(common_original)==common['original_sha256']and common['original_sha256']in captured[str((parent/'study_common.jl').resolve())],'Clock included canonical common binding drift')
    common_before='rawdir=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","clock-julia-"*model,"raw_timings")';common_after='rawdir=joinpath(ENV["STUDY_OUTPUT_DIR"],"private_clock")'
    fixture=' words=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(words)==1042'
    common_rebuilt=common_original.decode()
    for before,after in [(common_before,common_after),(fixture,fixture.replace('@__DIR__',json.dumps(str(parent))))]:
     need(common_rebuilt.count(before)==1,'Clock included-common path-only rule ambiguous');common_rebuilt=common_rebuilt.replace(before,after,1)
    need(common_rebuilt==common['text'],'Clock included-common changes exceed exact known paths')
    source_hashes[str((parent/'study_common.jl').resolve())]=common['original_sha256']
    changes=[('include(joinpath(@__DIR__,"..","study_common.jl"))','include('+json.dumps(common['copy'])+')')]
   rebuilt=original
   for before,after in changes:need(rebuilt.count(before)==1,'Clock exact path-only rule ambiguous');rebuilt=rebuilt.replace(before,after,1)
   need(rebuilt==p['text'],'Clock overlay changes more than exact approved path roots')
   inverse=rebuilt
   for before,after in reversed(changes):need(inverse.count(after)==1,'Clock inverse path-only rule ambiguous');inverse=inverse.replace(after,before,1)
   need(inverse.encode()==original_raw,'Clock exact inverse byte proof failed')
   overlay_checks.append({'benchmark':b,'model':m,'copy':p['copy'],'copy_sha256':p['copy_sha256'],'retained_plan_sha256':sha(planraw),'exact_path_only_transform_proved':True,'exact_inverse_byte_proved':True,'historical_copy_recreated':False})
'''
change('  def overlay(items):\n',clock+'  def overlay(items):\n')
change("   for e in items:\n    need(b=='amgmk'", "   for e in items:\n    if b=='clock':clock_overlay(e);continue\n    need(b=='amgmk'")
change("'retained_AMG_overlay_checks':overlay_checks,", "'retained_AMG_overlay_checks':[x for x in overlay_checks if x['benchmark']=='amgmk'],'retained_clock_overlay_checks':[x for x in overlay_checks if x['benchmark']=='clock'],")
(D/'merge_reviews_v4.original.py').write_bytes(B.read_bytes());(D/'merge_reviews_v5.py').write_text(v)
(D/'v4_to_v5.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),v.splitlines(True),fromfile=str(B),tofile='private/merge_reviews_v5.py')))
(D/'authoring.json').write_text(json.dumps({'base':str(B),'base_sha256':expected,'candidate_sha256':hashlib.sha256(v.encode()).hexdigest(),'mode':'SOURCE_AUTHORED_ONLY_NOT_IMPORTED_EXECUTED_TESTED_OR_APPLIED','scientific_note_required':'--root-scientific-review, exact path+SHA enters merge plan and receipt','counts':'exact equality of all seven expected_counts fields','commit_protocol':'unchanged journaled two ordinary same-filesystem renames, explicit non-atomic metadata EX; old v4 or new v5 marker blocks','clock_scope':'only Julia3 selected-plan retained text; independent exact transforms, source+copy captured pins, present originals, exact inverse bytes; no trust in changes array','no_new_count_claims':True},indent=2)+'\n')
print(hashlib.sha256(v.encode()).hexdigest())
