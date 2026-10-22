from pathlib import Path
O=Path(__file__).resolve().parent;s=Path('/tmp/roofline-completion-20261021/roi_membership_source_batch13_qr/write_companion.py').read_text();a=s.index('obs=[');b=s.index("(O/'companion_scientific_observations.json')",a)
obs="""obs=[{'benchmark':'s3d','observation':'The selected f64 chemistry-rate pipeline evaluates27stages over4096uniform synthetic cells with fixed1000temperature,1.0132e6pressure and unitmolecularweights.100wholepipeline launch/contextcompletion repeats are averaged.508sourceCPUformula values percell validate fullheterogeneous arrays outside the timer; this is not independent chemistry, coupledflow or timeintegration validation.','evidence':evidence('src/s3d-julia/study_common.jl',['fp64=ARGS','ht=fill','for rep in 1:R','heterogeneous_full_WDOT']),'scientific_intent':proof['scientific_intent']['s3d']},{'benchmark':'sa','observation':'The Int32recursiveDC3 suffix-array operator retains Thrust CABI allocation/sort/scan and host rankfeedback, alongside authored framework kernels. Timed constructions include assignment/H2D, recursive temporary allocation/free, customkernels, vendor sort/scan and D2H; outercasecontainer creation/deletion and independent permutation/finite-tail lexorder certification are outside the interval. Only final construction outputs pertrial are certified/archived, not every internalrepeat. Sourcefamily inclusion cannot establish sampled runtimephase or completevendorcountercoverage.','evidence':evidence('src/sa-julia/study_common.jl',['sa_sort(k','sa_scan(h','maxrank=sa_scalar','sa_new(0)','for it=1:repeats','elapsed=(time_ns()-t)','all_internal_iteration_outputs_archived']),'scientific_intent':proof['scientific_intent']['sa']},{'benchmark':'sc','observation':'The named Int32compaction timer contains oneGPUlaunch plus4CPUworker start/contextcompletion/join for each15staticpartition cases. Reset, padding checks, posttimer outputmerge and complete count/flag/guard oracle remain outside. This mixed host/device operator cannot be represented as wholeelapsedROI using its one devicefamily or interpreted as a causal framework/allocator comparison.','evidence':evidence('src/sc-julia/study_common.jl',['cut=trunc','function run(c)','ccall(startfn','ccall(waitfn','start=time_ns();run(c)','results==oracle']),'scientific_intent':proof['scientific_intent']['sc']}]
"""
s=s[:a]+obs+s[b:]
s=s.replace("# Revalidate ALL32 selection records including holds and exact parent INDEX gzip joins.","# Revalidate all32 selection records and exact source12 parent INDEX gzip joins.")
# Add exact reversible primarysourceencoding and selector-value proofs before final counts.
pos=s.index('matchrows=')
s=s[:pos]+'''encodingpath='/home/imo/HeCBench_julia/src/scan2-cuda/main.cu';d=proof['sources'][encodingpath];bb=Path(encodingpath).read_bytes();assert H(bb)=='b4ebfd20e21181df297456e8d162de10ca5b13e1850b383b0df7b022e78aa177'and d['snippet_encoding']=='latin-1'and d['snippet'].encode('latin-1')==bb
lineproofs=[]
for row in p['patches']:
 for op in row['stage_patches']:
  for ev in op['updates']['ROI_membership_proof']['evidence']+op['updates']['ROI_membership_proof']['timed_closure_boundaries']:
   if ev['path']!=encodingpath:continue
   ll=bb.decode('latin-1').splitlines(keepends=True)
   for q in ev['line_evidence']:
    i=q['line']-1;raw=ll[i].encode('latin-1');start=sum(len(x.encode('latin-1'))for x in ll[:i]);assert ll[i].rstrip('\\r\\n')==q['text']and bb[start:start+len(raw)]==raw;lineproofs.append({'benchmark':row['record_key']['benchmark'],'model':row['record_key']['model'],'stage_id':op['stage_id'],'line':i+1,'start_byte':start,'end_byte':start+len(raw),'encoding':'latin-1','text':ll[i]})
(O/'reversible_encoding_proof.json').write_text(json.dumps({'status':'PROPOSED_EXACT_ROOTNOTE_ENCODING_APPROVAL_REQUIRED_BEFORE_MERGE','proposed_approved_reversible_source_encodings':{encodingpath:{'sha256':H(bb),'encoding':'latin-1'}},'fullsource':d,'all_source_line_snippets_byteencode_offset_proofs':lineproofs,'fullbyte_inverse_equal':True,'genericfallback':False,'replace_ignore_skipheader':False,'other_badUTF8_header_unquoted':json.loads((O/'unquoted_source_scope.json').read_bytes())},indent=2)+'\\n')
corrections=[]
for row in p['patches']:
 for op in row['stage_patches']:
  u=op['updates']
  if u.get('regex_status','').startswith(('exact_source_ternary','exact_selected_CUDA_entry')):corrections.append({'record_key':row['record_key'],'stage_id':op['stage_id'],'expected_original_stage_canonical_sha256':op['expected_original_stage_canonical_sha256'],'new_regex':u['model_kernel_regex'],'basis':u['regex_status'],'exactcallsite':u['ROI_membership_proof']['exact_stage_callsite'],'evidence':u['ROI_membership_proof']['evidence']})
(O/'selector_correction_proof.json').write_text(json.dumps({'source_only':True,'no_genericdispatcher_identity_inferred':True,'corrections':corrections},indent=2)+'\\n')
''' + s[pos:]
(O/'write_companion.py').write_text(s)
