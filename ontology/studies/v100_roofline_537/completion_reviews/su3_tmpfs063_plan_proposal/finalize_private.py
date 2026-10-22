# CPU-only source/proof authoring. No proposed module imports/execution.
from pathlib import Path
import hashlib,json,os,shutil,datetime
D=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020')
P=Path('/tmp/roofline-completion-20261021/tmpfs_scratch_optin_proposal')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,v):(D/name).write_text(json.dumps(v,indent=2)+'\n')
def source_record(path,lo,hi,role):
 p=Path(path);lines=p.read_text().splitlines(keepends=True)
 return {'path':str(p),'sha256':sha(p),'line_start':lo,'line_end':hi,
  'exact_text':''.join(lines[lo-1:hi]),'role':role}
records=[]
common=R/'src/su3-julia/study_common.jl'
records.append(source_record(common,1,8,'Canonical fixture reads remain read-only via original @__DIR__; no674MB runtime copy.'))
records.append(source_record(common,32,54,'One synchronized untimed warmup then exactly five full trials; pre/post all raw device arrays, original three internal calls, all inputs/metadata/guards and full oracle. Only STUDY_OUTPUT_DIR redirects file destination.'))
for m,c in [('cuda_jl','su3-julia-cuda.jl'),('ka','su3-julia-KA'),('jacc','su3-julia-JACC')]:
 p=R/'src/su3-julia'/c/'main.jl';records.append(source_record(p,1,len(p.read_text().splitlines()),m+' canonical device body/entrypoint unchanged; original relative includes keep original source anchors.'))
native=RAW/'batch046-su3-native/su3/cuda/scratch/profile_source/study_native.cu'
records.append(source_record(native,24,24,'NATIVE HELD: unchanged source owns absolute old046 failure diagnostic. Success payloads are cwd-relative. Guard would deny an oldpath diagnostic write; no approval granted here.'))
records.append(source_record('/usr/bin/nvcc',1,3,'Existing compiler shell wrapper read-only; invokes /usr/lib/nvidia-cuda-toolkit/bin/nvcc. Native is held and no compiler is executed.'))
save('source_role_proof.json',{'review':'Read-only source/static review only, no new GPU or raw numerical audit. Existing accepted result/source joins retained.',
 'sources':records,'symbolic_tmpfs_namespace':'/dev/shm/hecbench-v100-uid10384-su3_batch063_root_review_pending',
 'Julia_outputs':'Five profile_outputs/su3_trial[1..5].bin at1,347,849,664bytes each, existing env-driven path only.',
 'native_status':'held_absolute_failure_diagnostic_literal_not_relocatable_by_plan_only',
 'native_next_action':'Separate future failure-path-only source-copy intervention and narrowly extended normalization review, not authorized by this packet.',
 'all_four_skip_logic':'Follower snapshots all four canonical statuses; native never executed. Collector skip-qualified-selected applies independently to eligible Julia3.',
 'legacy_method':'Keep actual2022.4.1.0 method, full27metrics, exact existing selector/replay/cache/clock and guards; no2025compatibility claim.'})
# Future tool copies are byte-identical to the already delivered proposal.
alt=D/'future_tools';alt.mkdir(exist_ok=True)
for original,newname in [(P/'proposed/tools/v100_roofline_collect.py','v100_roofline_collect_tmpfs.py'),(P/'proposed/tools/v100_tmpfs_scratch.py','v100_tmpfs_scratch.py')]:
 (alt/newname).write_bytes(original.read_bytes())
tools={str(R/'tools/v100_roofline_collect_tmpfs.py'):sha(alt/'v100_roofline_collect_tmpfs.py'),
 str(R/'tools/v100_tmpfs_scratch.py'):sha(alt/'v100_tmpfs_scratch.py'),
 str(R/'tools/v100_roofline_collect.py'):sha(R/'tools/v100_roofline_collect.py'),
 str(R/'tools/v100_profiler_method.py'):sha(R/'tools/v100_profiler_method.py'),
 str(R/'tools/v100_roofline_classify.py'):sha(R/'tools/v100_roofline_classify.py'),
 str(R/'tools/v100_profile_write_guard.py'):sha(R/'tools/v100_profile_write_guard.py')}
legacy_paths=['/usr/bin/ncu','/usr/lib/nsight-compute/ncu','/usr/lib/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu',
 str(R/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json')]
legacy={path:sha(path)for path in legacy_paths}
review=json.loads((D/'root_review_DRAFT_NOT_APPROVED.json').read_text())
for row in review['models'].values():row['read_only_bindings'].update(tools);row['read_only_bindings'].update(legacy)
save('root_review_DRAFT_NOT_APPROVED.json',review)
predecessors=[]
for pid,start,reason in [(1160651,'76203887','Whole historical SU3 original controller'),(1193331,'76362961','Whole historical SU3 native followup'),(1283663,'76808696','Whole046–049 large-space follower, including Perlin/ZMDDFT; do not infer child completion from absent individual slot')]:
 p=Path('/proc')/str(pid);argv=None;actual=None
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();actual=fields[19]
  if actual==start:
   assert p.stat().st_uid==os.getuid();argv=[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x]
 except FileNotFoundError:pass
 predecessors.append({'pid':pid,'start_ticks':start,'uid':os.getuid(),'argv':argv,'observed_start_ticks':actual,'reason':reason})
before=json.loads((D/'before_bindings.json').read_text());after={}
for path,digest in before['original_plans'].items():
 after[path]=sha(path);assert after[path]==digest
for path,digest in before['canonical_source_bindings'].items():
 after[path]=sha(path);assert after[path]==digest
save('after_bindings.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_original_and_source_bindings_unchanged':True,'bindings':after,'snapshot':'Sequential source rehashes; not simultaneous and not a new verification of historical full output files.'})
packet={str(p):sha(p)for p in D.rglob('*')if p.is_file()and p.name not in ('launch_review_DRAFT_NOT_APPROVED.json','DELIVERY.json','usage_receipt.json','controller_receipt.json')}
launch={'schema_version':1,'root_reviewed':False,'root_authorizes_fresh_tmpfs_staging':False,
 'controller_sha256':sha(D/'wait_then_collect.py'),'source_root':str(R),'raw_root':str(RAW),
 'future_plan_root':str(RAW/'batch063-su3-tmpfs'),'owned_tmpfs_root':'/dev/shm/hecbench-v100-uid10384-su3_batch063_root_review_pending',
 'python':'/usr/bin/python3','maximum_collector_invocations':1,'eligible_models':['cuda_jl','ka','jacc'],'native_held':True,
 'predecessors':predecessors,'packet_bindings':packet,'future_tool_bindings':tools,
 'legacy_profiler_wrapper':'/usr/lib/nsight-compute/ncu','legacy_method_bindings':legacy,
 'storage_review_required_plan_sha256':{k:v['fresh_plan_sha256']for k,v in review['models'].items()},
 'root_review_requirements':['Source-read follower and exact0226/e2ea proposal; deploy as additional collector filename only, preserve C21 pending059060/062.',
 'New controller/supervisor frontier pins additional tools and exact both approved review documents before any external staging.',
 'No native execution; held source failure root remains historical and cannot be approved by this path-only plan authority.',
 'Tmpfs non-durable until archival, no primary raw deletion or timing replacement; all five actual full payloads/checks retained.',
 'No executing --version/compiler/target/collector by author. Symbolic paths only; no /dev/shm writes.' ]}
save('launch_review_DRAFT_NOT_APPROVED.json',launch)
print(json.dumps({'collector_sha256':tools[str(R/'tools/v100_roofline_collect_tmpfs.py')],'helper_sha256':tools[str(R/'tools/v100_tmpfs_scratch.py')],'follower_sha256':launch['controller_sha256'],'root_review_draft_sha256':sha(D/'root_review_DRAFT_NOT_APPROVED.json'),'launch_review_draft_sha256':sha(D/'launch_review_DRAFT_NOT_APPROVED.json'),'legacy_which_ncu_now':shutil.which('ncu')}))
