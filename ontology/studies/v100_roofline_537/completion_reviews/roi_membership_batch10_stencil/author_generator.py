from pathlib import Path
import json
O=Path(__file__).parent;s=Path('/tmp/roofline-completion-20261021/roi_membership_batch9_stencil/prepare_proposals.py').read_text();a=s.index('N=');z=s.index('\ndef sha',a);s=s[:a]+'N=set('+repr(json.loads((O/'selection.json').read_text())['names'])+')'+s[z:]
why={
'mandelbrot':'Selected study calls mandel through explicit closure inside one synchronized timer per named image; header original-main launch is dormant. Julia explicit factory closure enters same per-image launch+completion timer. Source Float32 complex recurrence with explicit fusion convention and Int32 escape counts, not image quality or continuum fractal validation. Input/parameter capture, setup and full oracle after/before timer. First captured invocation may be warmup.',
'marchingCubes':'Only generatingTriangles/mc_stage5 closure is timed. First four stages minmax/active-cube compaction are invoked in preprocess! and separately reported untimed_preprocessing_seconds before start. Native study similarly preprocesses before timed generation closure; dormant original main aliases excluded. Julia factory maps kernel1..5 explicitly to corresponding ops. Source Float32 field/interpolation, UInt table/index/count and partial coordinate/checksum output; full mesh arrays not produced. No all-pipeline timed-ROI claim; profiling firstperfamily may sample preprocessing.',
'mask':'Native six switch cases and Julia Val(mode) dispatcher called for six mask operators across two fixtures in repeat timer. Selected sequence/window/upper/lower/diagonal predicates operate Int32 values/indices, not scalarFP. Warmup and CPU independent predicate/oracle checks outside. Legacy original main host loops dormant. Generic callback maps to explicit mask kernel with mode; vendor placeholder is not proof of emitted library family.',
'match':'Native selected switch directly launches ten matching kernels per two cases inside three-repeat timer. Julia explicit k-dispatch maps same ten callbacks/KAhandles; repeated dot-product argmax and tie-order fixture semantics source-bound. Float32 input/product/reduction with Int32 winning IDs, not general nearest-neighbor or arbitrary tie optimality claim. Upload/reset and score/index oracle outside; legacy original/study_source standalone main dormant.',
'matern':'Selected native and Julia invoke matern kernel across two weight modes and seven scales in five-repeat synchronized timer; first direct native invocation warmup outside. Source Float32 distance/SFU weighted kernel sum with Float64 independent oracle, positive-weight monotonicity/bounds tested outside. Not covariance-matrix SPD/general statistical model validation; original main diagnostics uncalled.',
'matrix-rotate':'Selected N1024/repeats10: native first direct launch warmup excluded; repeat-loop in-place rotate launch timed. Julia explicit rotate callback in common repeat timer. Float32 storage moved by integer ring/cycle indexing; no ordinary FP arithmetic implied by stored format. Input reset, CPU rotation oracle and readback outside timer.',
'matrixT':'Native study uses nine function-pointer entries including copies, partial transposes and complete transposes, each inside own repeat timer summed; unrelated original main launch candidate alone insufficient for inclusion. Julia factory tuple indexkind1..9 binds explicit callbacks/KAhandles/JACCmode dispatch in same nine per-operation timers. Float32 payload movement with integer indexing; source vector/shared layouts do not imply FP arithmetic. Original host main/benchmark wrappers uncalled; no full-image numerical method claimed.',
'maxpool3d':'Native study invokes maxpool3d through closure inside ten-repeat timer; original standalone main dormant. Julia prepare binds explicit pool callback and OwnedStudy run_trials times same repeats after reset. Named256x256x16 shape with2x2x2 max selection, Float32 storage/comparisons plus integer offsets. Warmup/reset/transfers/CPUoracle outside; not full neural-network training.',
'mcmd':'Common native mcmd_run host controller calls force_kernel and authored Julia CABI callback into explicit mcmd_launch. Native force_dispatch launches calculateForceKernel; archived original force_kernel wrapper uncalled. Timer belongs full original11force-call MC/MD workflow including CPU controller/allocation/H2D/D2H/free; instrumented per-force input/output observer durations subtracted, not GPU completion. Float64 pform1 LJ+Ewald finite fixture; no arbitrary potential modes or full experimental physics certification. Generic vendor candidate not emitted-kernel proof.',
'md':'Selected native/Julia run three129-particle force-only fixtures20times inside synchronized timer; warmup direct native call outside. Float32 cutoff neighbor-list force with source LJ constants, independent Float64 pairwise/potential-gradient and symmetry checks outside. Coincident/exact-cutoff controls preserved; no integrated MD trajectory. Original host aliases uncalled; vendor placeholder unresolved.',
'mdh':'Selected native invokes three vectorized potential kernels and Julia run!Val modes0/1/2 inside per-mode10-repeat timers summed. Float32 four-lane storage/accumulation, shared atom tiles, screened-charge exp/sqrt; vector payload not FP16/tensor proof. Atom129/grid1024 fixture setup and Float64 screened-potential oracle outside. Original main aliases dormant.',
'meanshift':'Captured argv tiled selects only tiled native/KA/JACC callback branch; base branch explicitly excluded. Factory executes1000iteration ping-pong launches with periteration completion, timed total divided1000; reset and converged-state oracle outside. Float32 Gaussian3D mean shift with distance threshold60 and denominator32,10000points. No sampled iteration/convergence speed or causal cache/shared claim.',
'medianfilter':'Selected ckMedian launch and Julia explicit callback execute100repeat timer after warmup. Source packed UInt32 RGB bytes, eight binary-search steps and zero-extended3x3 neighborhood; true rank median check allows source one-byte approximation difference, alpha0 contract retained. Integer/bitwise image operator, not Float32 despite historical labels; oracle/readback outside.',
'memtest':'Source timed pipeline init/move/check executes three repeats with per-stage completion; separate intentional fault/check probe before primary timer is excluded. Julia Symbol dispatch explicitly picks init!,move!,check!, rather than generic JACC f identity by name alone. UInt32 patterns/movement plus UInt64 address/error bookkeeping,2MiB scoped test5; injected-fault and fullpayload checks outside. Not all memory tests or hardware reliability certificate.',
'merge':'Captured mode float64; selected native diagonal-path then window merge and Julia factory corresponding two callbacks execute inside ten-repeat synchronized timer.100000+100000 positive sorted inputs, Float64 storage/comparison and integer path searches; no ordinary floating arithmetic denominator inferred. Resets/path/outputoracle and guard checks outside. Original standalone host paths uncalled.',
'mergeVS':'Selected StudyFixture execute invokes original convert_vertical_slash_indexes host wrapper that launches mergehead kernel; Julia exact callback maps same metadata operator. Fourfixture/causal-mode paths have separate repeat timers summed. Int32/Int64 sparse attention metadata/index merge only, not actual attention/softmax arithmetic. Setup, reset, actualinput/guards/CPUoracle outside; sparse metadata interval semantics do not certify attention read validity.'}
a=s.index('why=');z=s.index('\nlock=',a);s=s[:a]+'why='+repr(why)+s[z:]
a=s.index('  overlays=[]');z=s.index('  ev=[];',a)
s=s[:a]+'''  overlays=[]
  if plan.get('capture_overlay_provenance',{}).get('patches'):
   withheld.append({'benchmark':b,'model':m,'reason':'Executed overlay requires separate inverse-source proof; not promoted'});continue
  # Quote only actual prospectively pinned source bytes, including helper/controller paths.
  paths=set()
  for k in plan['execution_bindings']:
   p=Path(k)
   if p.suffix in ('.jl','.cu','.cuh','.h','.hpp','.cpp','.c') and (str(base)+'/' in str(p) or '/src/'+b+'-cuda/' in str(p) or str(p).endswith('/src/_portable_study/OwnedStudy.jl')):
    paths.add(p.resolve())
  if not paths:
   withheld.append({'benchmark':b,'model':m,'reason':'No pinned actual source text available'});continue
'''+s[z:]
a=s.index("   path=Path(s['source_callsite']['path']);legacy=");z=s.index("   updates['ROI_membership_proof']=",a)
s=s[:a]+'''   if not s.get('source_callsite'):
    continue # vendor placeholders stay unresolved, never inferred from API name
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   excluded=None
   if m=='cuda':
    if b=='matrixT':excluded='uninvoked_original_main_without_exact_function_pointer_candidate_join'
    elif b=='marchingCubes':excluded='untimed_scientific_preprocessing'if si<4 else 'uninvoked_legacy_main'if si>=5 else None
    elif b=='mask':excluded='uninvoked_legacy_main'if si>=6 else None
    elif b=='match':excluded='uninvoked_original_or_archived_standalone_main'if si>=10 else None
    elif b=='matern':excluded='untimed_warmup'if si==0 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='matrix-rotate':excluded='untimed_warmup'if si==0 else None
    elif b=='mcmd':excluded='uninvoked_archived_force_wrapper'if si==1 else None
    elif b=='md':excluded='untimed_warmup'if si==0 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='mdh':excluded='uninvoked_legacy_main'if si>=3 else None
    elif b=='meanshift':excluded='argv_unselected_base_branch'if si==1 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='medianfilter':excluded='untimed_warmup'if si==0 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='memtest':excluded='untimed_intentional_fault_audit'if si==3 else None
    elif b=='merge':excluded='uninvoked_legacy_main'if si>=2 else None
    elif legacy and b not in ['mergeVS']:excluded='uninvoked_legacy_main'
   else:
    if b=='marchingCubes'and si<4:excluded='untimed_scientific_preprocessing'
    if b=='mask'and m=='cuda_jl'and si==6:excluded='uninvoked_legacy_main'
    if b in ['maxpool3d','md','medianfilter']and m=='cuda_jl'and si>=1:excluded='uninvoked_legacy_main'
    if b=='matrixT'and m=='cuda_jl'and si>=1:excluded='uninvoked_legacy_host_wrapper'
    if b=='meanshift'and ((m=='ka'and si==0)or(m=='jacc'and si==1)):excluded='argv_unselected_base_branch'
   kind=excluded or 'timed_semantic_family';included=excluded is None
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_untimed_audit_exclusion_proved'if excluded=='untimed_intentional_fault_audit'else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_'+excluded)
'''+s[z:]
# Skip unchanged null vendor stages while retaining exact original index layout.
s=s.replace("revised=[dict(s,**patches[i]['updates'])for i,s in enumerate(r['stages'])]","revised=[dict(s)for s in r['stages']]\n  for p in patches:revised[p['stage_index']].update(p['updates'])")
s=s.replace('six named cases','sixteen m-prefix named cases').replace('batch9','batch10')
(O/'prepare_proposals.py').write_text(s)
