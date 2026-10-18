#!/usr/bin/env python3
"""Export the full census without converting unknowns into successes or failures."""
from pathlib import Path
from study_evidence import effective_config
from collections import Counter
import argparse, datetime, hashlib, json, re, subprocess, time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.porting-state/portable_julia'
PAPER=ROOT/'2026-PASC-Hecbench-Julia-paper'
OUT=ROOT/'ontology/studies/julia_abstractions_300'
MODELS={'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}
CODES=['B1','B2','B3','B4','R1','R2','R3','N1','N2','N3','S1','S2','P1','T1']

def read(p,default=None):
    # Missing evidence is unknown; malformed existing evidence must not silently erase it.
    for attempt in range(4):
        try:return json.loads(p.read_text())
        except FileNotFoundError:return default
        except json.JSONDecodeError:
            if attempt==3:raise
            time.sleep(.05)

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def jsonlines(p):
    if not p.exists():return []
    for attempt in range(4):
        try:
            return [(i,json.loads(line)) for i,line in enumerate(p.read_text().splitlines(),1) if line.strip()]
        except json.JSONDecodeError:
            if attempt==3:raise
            time.sleep(.05) # A concurrent append may have exposed an incomplete line.


def usage_windows(benchmark,model):
    windows=[]
    for f in sorted((STATE/'usage_boundaries_300'/benchmark).glob(model+'_*.jsonl')):
        events=[e for _,e in jsonlines(f)]
        for e in events:
            if e.get('boundary') not in ['end','abandon']:continue
            begin=next((x for x in events if x.get('event_id')==e.get('begin_event_id')),None)
            windows.append({k:e.get(k) for k in ['event_id','begin_event_id','agent','implementation_model','phase','boundary','recorded_at','usage_event_timestamp','elapsed_usage_event_start','delta_usage','positive_counter_span','source_identity_sha256','marker_lag_seconds','accounting_level','source_token_event_sha256','meter_driver_sha256','note','measurement_status','scope']} | {'begin_marker_lag_seconds':(begin.get('marker_lag_seconds') if begin.get('marker_lag_seconds') is not None else (datetime.datetime.fromisoformat(begin['recorded_at'])-datetime.datetime.fromisoformat(begin['usage_event_timestamp'].replace('Z','+00:00'))).total_seconds()) if begin else None,'evidence_path':str(f.relative_to(ROOT)),'evidence_sha256':digest(f),'counter_span_interpretation':'Observed counter difference; alignment and interleaving prevent exclusive billed-cost claims. Zero/no-new-event spans cannot imply zero translation effort.'})
    for w in windows:
        if w.get('marker_lag_seconds') is None and w.get('recorded_at') and w.get('usage_event_timestamp'):
            w['marker_lag_seconds']=(datetime.datetime.fromisoformat(w['recorded_at'])-datetime.datetime.fromisoformat(w['usage_event_timestamp'].replace('Z','+00:00'))).total_seconds()
            w['marker_lag_derived_from_archived_timestamps']=True
    return windows

# Exact source-line reviews retain raw events while correcting attribution and
# grouping duplicate descriptions. Never deduplicate by textual similarity.
def load_event_reviews():
    path=ROOT/'ontology/studies/julia_abstractions_300/event_review_overrides.json';document=read(path,{})
    reviews={}
    for review in document.get('entries',[]):
        source=ROOT/review['source_path'];number=review['line']
        lines=source.read_text().splitlines()
        if type(number) is not int or number<1 or number>len(lines):
            raise ValueError('Event review line missing: '+str(source))
        if hashlib.sha256(lines[number-1].strip().encode()).hexdigest()!=review['source_line_sha256']:
            raise ValueError('Event review source hash mismatch: '+str(source)+':'+str(number))
        if review['canonical_origin_model'] not in [*MODELS,'cuda','shared_infrastructure']:
            raise ValueError('Unknown reviewed origin')
        key=(review['source_path'],number)
        if key in reviews:raise ValueError('Duplicate event review selector')
        reviews[key]=review
    return reviews
def apply_event_review(event,reviews):
    review=reviews.get((event['evidence_path'],event['evidence_line']))
    if review:
        event['origin_model']=review['canonical_origin_model']
        if 'is_failure' in review:
            if type(review['is_failure']) is not bool:raise ValueError('Review failure flag must be boolean')
            event['is_failure']=review['is_failure']
        event['canonical_observation_id']=review['canonical_observation_id']
        event['exact_source_line_review']=review
    return event

# Preserve equivalent historical counter spellings and expose conflicts rather
# than silently interpreting absent canonical keys as absent evidence.
EFFORT_ALIASES={
    'candidate_attempts':['candidate_attempts','translation_candidate_attempts'],
    'compiler_checks':['compiler_checks','compiler_check_invocations'],
    'compiler_feedback_repairs':['compiler_feedback_repairs','compiler_feedback_repair_rounds'],
    'runtime_repairs':['runtime_repairs','runtime_repair_rounds','runtime_feedback_repairs'],
    'numerical_repairs':['numerical_repairs','numerical_repair_rounds','numerical_feedback_repairs','translation_numerical_repairs']}
def normalize_effort(manifest):
    values={};provenance={}
    for field,aliases in EFFORT_ALIASES.items():
        raw={name:manifest[name] for name in aliases if name in manifest}
        explicit={v for v in raw.values() if type(v) is int and v>=0}
        values[field]=next(iter(explicit)) if len(explicit)==1 else None
        provenance[field]={'raw_alias_values':raw,'status':'conflict' if len(explicit)>1 else 'recorded' if explicit else 'missing_or_noninteger'}
    values['translation_tokens']=manifest.get('translation_tokens')
    return values,provenance

# Candidate families are an explicitly provisional sorting aid, not reviewed provenance.
FAMILIES={
'pde_continuum':'adv axhelm burger cooling divergence fdtd3d fluidSim heat heat2d hotspot iso2dfd jacobi laplace laplace3d lid-driven-cavity rtm8 s3d stencil1d stencil3d sw4ck vanGenuchten damage tissue',
'particle_molecular':'bh haccmk md mdh particle-diffusion particles',
'signal_image':'affine asmooth bilateral boxfilter cbsfil colorwheel convolution3D convolutionSeparable dct8x8 degrid fft fwt gabor gamma-correction gaussian grrt henry histogram hogbom hwt1d lombscargle matrix-rotate medianfilter morphology overlay resnet-kernels seam-carving slit sobel sosfil upsample winograd zoom',
'linear_algebra':'axpby babelstream blas-dot blas-fp4gemm blas-fp8gemm blas-gemm blas-gemmEx blas-gemmStridedBatched blas-groupgemm blas-mxfp8gemm bspline-vgh ccsd-trpdrv chemv cross determinant eigenvalue ge-spmm geam gels gemv hadamard inversek2j jacobian lud ludb matrixT norm2 qrg simpleSpmv spaxpby spd2s thomas triad',
'stochastic_statistics':'black-scholes chi2 entropy feynman-kac fpc fresnel ga gc gd hausdorff hellinger hmm ising jaccard keogh knn libor matern mcpr meanshift minkowski mrg32k3a multinomial mtf perplexity pso rng-wallace sobol urng',
'machine_learning':'accuracy adam adamw addBiasQKV addBiasResidualLayerNorm attention attentionMergeState attentionMultiHead attentionMultiHeadKVCache background-subtract bgmv channelShuffle channelSum compute-score crossEntropy dense-embedding fma geglu gelu glu goulash groupnorm gru gru2 lif linearprobing logprob mask maxpool3d mlp moe moe-align moe-sum mrc muon mxfp4 nlll nms nonzero pool qkv quant3MatMul quantBnB quantVLLM relu reshapeKVCache score scel segment-reduce silu softmax softmax-fused softmax-online split ssm stddev storeKVCache tensorAccessor tensorT topk vol2col',
'graph_irregular':'bfs btree epistasis floydwarshall is jenkins-hash local-ht marchingCubes nqueen nw page-rank pnpoly pns qem rainflow surfel tsa wordcount wyllie',
'discrete_crypto':'aes base64e bitpacking bitpermute chacha20 ecdh keccaktreehash md5hash present secp256k1',
'hardware_data_movement':'aligned-types atomicAggregate atomicCAS atomicCost atomicIntrinsics atomicPerf atomicReduction bscan blockAccess blockexchange concurrentKernels dp4a f16atomic f16max f8cast graphExecution intrinsics-cast layout lfib4 memtest nosync openmp overlap pitch prefetch randomAccess ring shmembench shuffle streamPriority streamUM threadcpy threadfence warpexchange zerocopy',
'domain_neutral_primitives':'adjacent aidw asta bezier-surface bincount bitonic-sort bwt clink concat coordinates daphne fsm langford lebesgue match merge mergeVS minmax mr p4 projectile radixsort radixsort2 reverse rfs rle romberg s8n scan scan2 scan3 scatter scatterAdd scatterThrust sort sortKV spm unfold vadd wedford zeropoint'
}
LOOKUP={b:k for k,v in FAMILIES.items() for b in v.split()}

def scan_source(path):
    if not path.exists():return None
    text=path.read_text(errors='replace')
    patterns={'cuda_kernel_launch':r'@cuda|\bcudacall\b','ka_kernel':r'@kernel',
              'jacc_parallel_loop':r'JACC\.(?:parallel_for|@parallel_for)',
              'local_storage':r'@localmem|@cuStaticSharedMem|shared_array',
              'barrier':r'@synchronize|sync_workgroup|sync_threads',
              'atomic':r'atomic|Atomix|@atomic','vendor_library':r'CUBLAS|CURAND|CUFFT|cuBLAS|cuRAND|curand|cublas|cufft',
              'explicit_zero_dynamic_shared':r'shmem_size\s*=\s*0'}
    return {'method':'syntax-pattern evidence only; not a reviewed algorithm or bottleneck',
            'path':str(path.relative_to(ROOT)),'sha256':digest(path),
            'syntax_features':{k:[i for i,line in enumerate(text.splitlines(),1) if re.search(p,line)][:12]
                               for k,p in patterns.items()},
            'placeholder_signals':{'include_alias':bool(re.search(r'include\([\"\']\.\./',text)),
                                   'tiny_source':len(text.splitlines())<10}}

def normalize_traits(t):
    if not t:return None
    def pick(*names):
        return next((t[n] for n in names if t.get(n) is not None),None)
    precision=t.get('precision',{})
    if not isinstance(precision,dict):precision={}
    return {
        'algorithm':pick('actual_algorithm','algorithm','algorithm_and_mapping'),
        'implementation_class':pick('implementation_class'),
        'declared_environment':pick('environment','environment_id'),
        'package_versions':pick('package_versions'),
        'julia_language_version':pick('julia_language_version'),
        'work_item_mapping':pick('work_item_mapping','mapping','algorithm_and_mapping','thread_mapping'),
        'layout':pick('layout','data_layout','memory_layout','storage_layout'),
        'storage_precision':pick('storage_precision') or precision.get('storage'),
        'compute_precision':pick('compute_precision') or precision.get('compute'),
        'timed_compute_precision':pick('timed_compute_precision'),
        'untimed_device_compute_precision':pick('untimed_device_compute_precision'),
        'host_oracle_precision':pick('host_oracle_precision'),
        'artifact_encoding':pick('artifact_encoding'),
        'memory_hierarchy_and_access':pick('memory_hierarchy','memory_access','memory'),
        'synchronization':pick('synchronization'),
        'atomics_and_reductions':pick('atomic_operations','atomics','reductions'),
        'resource_policy':pick('resource_policy','resource_geometry'),
        'launch_geometry':pick('launch_geometry','resource_geometry'),
        'library_backend_boundary':pick('library_boundary','backend_boundary','backend_extensions','actual_backend'),
        'host_control_boundary':pick('host_control_boundary','host_device_boundary'),
        'support_boundary':pick('support_boundary'),
        'native_to_model_mapping':pick('native_to_model_mapping'),
        'timing_scope':pick('timing_scope'),
        'fidelity_status_and_scope':pick('fidelity_scope','fidelity_status','status'),
        'source_review_evidence':pick('source_evidence','provenance','source_revision','source_sha256'),
        'null_interpretation':'Not recorded in this source review; never assume absent feature or zero resource use.'}

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--population',type=int,choices=[300,447],default=300);args=ap.parse_args();population=args.population
    OUT=ROOT/f'ontology/studies/julia_abstractions_{population}'
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    event_reviews=load_event_reviews()
    baseline=read(STATE/f'baseline_{population}.json');cohort=read(STATE/f'batch_{population}.json')
    selected=set(cohort['benchmarks']);base={x['bench']:x for x in baseline['benchmarks']}
    adjud=read(PAPER/'research/data/adjudications.json',{});disp=read(PAPER/'research/data/configuration_dispositions.json',{})
    latest={}
    for p in sorted((STATE/'runs').glob('*/result.json')):
        d=read(p,{})
        if d.get('benchmark') not in selected or adjud.get(d.get('run_id'),{}).get('exclude_from_performance'):continue
        c=effective_config(p,read(p.with_name('config.json'),{}));cid=c.get('configuration_id')
        if disp.get(cid,{}).get('role',c.get('configuration_role','primary')) not in ['primary','corrected_reference_secondary']:continue
        for model in MODELS:
            if model=='cuda_jl':
                good=d.get('reference_agreement') and all(d.get('models',{}).get(m,{}).get('complete') and d.get('models',{}).get(m,{}).get('status')=='native_check_pass' for m in ['cuda','cuda_jl'])
            else:
                execution_only=c.get('numerical_agreement_kind')=='no_numerical_outputs_execution_only'
                allowed_statuses=('execution_only_pass',) if execution_only else ('direct_artifact_pass','numeric_summary_pass')
                trial_status='execution_only_pass' if execution_only else 'pass'
                good=d.get('variant_verified',{}).get(model) and d.get('reference_agreement') and all(
                    d.get('comparisons',{}).get(model,{}).get(ref,{}).get('status') in allowed_statuses
                    and len(d.get('comparisons',{}).get(model,{}).get(ref,{}).get('trials',[]))==5
                    and all(x.get('status')==trial_status for x in d['comparisons'][model][ref]['trials'])
                    for ref in ('cuda','cuda_jl'))
            if good:latest[(d['benchmark'],model,cid)]=(d,c,p)
    domains=read(STATE/'scientific_domains_300.json',{})
    statuses=[];all_events=[];all_reference_events=[];all_infrastructure_events=[];domain_rows=[]
    for b in cohort['benchmarks']:
        traits=read(STATE/'traits'/(b+'.json'),{})
        # A completed operator audit supersedes an earlier header-level review.
        audited_intent=traits.get('scientific_intent')
        classification=traits.get('scientific_classification')
        if not audited_intent and classification and 'provisional' not in classification.get('review_status','provisional'):
            audited_intent={**classification,
                'primary_family':classification['broad_family'],
                'scientific_task':classification.get('mathematical_operator'),
                'evidence':classification.get('source_evidence',[]),
                'classification_method':'Source-reviewed scientific_classification fields normalized from local traits; no name-derived promotion.',
                'classification_source':str((STATE/'traits'/(b+'.json')).relative_to(ROOT))}
        if audited_intent and 'provisional' not in audited_intent.get('review_status','provisional'):
            intent=audited_intent
        else:
            intent=domains.get(b) or audited_intent
        if not intent:intent={'primary_family':LOOKUP.get(b,'unclassified'), 'review_status':'provisional_category_requires_source_review','scientific_task':None,'community_use_case':None,'evidence':[],'classification_method':'Curated name/motif grouping to organize source review; not validated application provenance.'}
        domain={'schema_version':2,'benchmark_id':b,'scientific_intent':intent,
                'base_ontology':{'path':base[b].get('ontology_path'),'frozen_sha256':base[b].get('ontology_sha256'),'status':'inferred base; not measured evidence'},
                'historical_pass_method':base[b].get('method'),'historical_pass_is_strengthened_verification':False,'shared_observed_usage_windows':usage_windows(b,'shared')}
        witness_path=STATE/'scientific_witnesses_300'/(b+'.json')
        witness=read(witness_path)
        if witness:
            domain['scientific_witness_evidence']={'path':str(witness_path.relative_to(ROOT)),'sha256':digest(witness_path),
                'summary':{k:v for k,v in witness.items() if k!='checks'},
                'model_trial_summary':[{'model':x.get('model'),'trial':x.get('trial'),'pass':x.get('pass'),'artifact_sha256':x.get('artifact_sha256')} for x in witness.get('checks',[])],
                'interpretation':'Additional scoped mathematical/invariant evidence; complements full dual-reference checks, not a proof over all inputs.'}
        write(OUT/'domains'/(b+'.json'),domain);domain_rows.append(domain)
        for line,event in jsonlines(STATE/'failure_events_300'/(b+'.jsonl')):
            if event.get('implementation_model') in ['cuda','native_cuda']:
                all_reference_events.append({'benchmark_id':b,'evidence_path':str((STATE/'failure_events_300'/(b+'.jsonl')).relative_to(ROOT)),'evidence_line':line,'raw_event':event})
        for line,event in jsonlines(STATE/'failure_events_300'/(b+'.jsonl')):
            if event.get('implementation_model') in ['shared','shared_infrastructure','infrastructure']:
                all_infrastructure_events.append({'benchmark_id':b,'evidence_path':str((STATE/'failure_events_300'/(b+'.jsonl')).relative_to(ROOT)),'evidence_line':line,'raw_event':event})
        for model,suffix in MODELS.items():
            folder=ROOT/'src'/(b+'-julia');path=folder/(b+'-julia-'+suffix)/'main.jl'
            if model=='cuda_jl' and not path.exists():path=folder/'main.jl'
            log_root=ROOT/'.porting-state/porting_logs'
            aliases=['cuda','cuda.jl','cuda_jl'] if model=='cuda_jl' else [model]
            manifest_candidates=[log_root/(b+'-julia-'+alias)/'manifest.json' for alias in aliases if (log_root/(b+'-julia-'+alias)/'manifest.json').is_file()]
            # Retain the canonical legacy CUDA.jl log as a separate candidate.
            # Fresh reference preparations may use this path; absent counters stay unknown.
            if model=='cuda_jl' and (log_root/(b+'-julia')/'manifest.json').is_file():
                manifest_candidates.append(log_root/(b+'-julia')/'manifest.json')
            # Prefer a current source-bound record, then the latest explicit record.
            # Retain every alias candidate; never merge incompatible histories silently.
            selected_manifest=max(manifest_candidates,key=lambda q:(bool(digest(path) and read(q,{}).get('source_sha256')==digest(path)),q.stat().st_mtime_ns)) if manifest_candidates else log_root/(b+'-julia-'+aliases[0])/'manifest.json'
            log=selected_manifest.parent
            manifest=read(selected_manifest,{})
            normalized_effort,effort_field_provenance=normalize_effort(manifest)
            evidence=[(cid,v) for (bench,m,cid),v in latest.items() if bench==b and m==model]
            scan=scan_source(path)
            status='verified_named_configuration' if evidence else 'in_progress' if manifest or (model!='cuda_jl' and path.exists()) else ('existing_cuda_jl_not_reaudited' if base[b].get('method')=='existing_cuda_jl_entrypoint_only' else 'historical_pass_not_reaudited') if model=='cuda_jl' else 'not_started'
            explicit=read(STATE/'status_300'/b/(model+'.json'),{})
            if evidence and any(v[1].get('verification_contract_revised') for _,v in evidence):status='conditionally_verified_revised_acceptance_contract'
            if evidence and any(v[1].get('native_reference_repaired') for _,v in evidence):status='conditionally_verified_native_reference_repaired'
            if evidence and any(v[1].get('native_behavior_known_scientific_defect') for _,v in evidence):status='verified_native_behavior_known_scientific_defect'
            if evidence and all(v[1].get('numerical_agreement_kind')=='no_numerical_outputs_execution_only' for _,v in evidence):status='execution_only_verified'
            if explicit.get('status') and not evidence:status=explicit['status']
            events=[]
            sources=[(log/'compiler_attempts.jsonl',False),(STATE/'failure_events_300'/(b+'.jsonl'),True)]
            normalized_links=set()
            for _,event in jsonlines(STATE/'failure_events_300'/(b+'.jsonl')):
                if event.get('implementation_model',event.get('model'))!=model or not event.get('preserved_event_source'):continue
                check_ids=event.get('preserved_event_check_ids') or ([event['check_id']] if event.get('check_id') is not None else [])
                normalized_links.update((event['preserved_event_source'],check_id) for check_id in check_ids)
            for file,shared in sources:
                for line,e in jsonlines(file):
                    if not shared and (str(file.relative_to(ROOT)),e.get('compiler_check',e.get('check_id'))) in normalized_links:continue
                    em=e.get('implementation_model',e.get('model'))
                    if shared and em not in [model,'julia-'+model, suffix]:continue
                    code=e.get('taxonomy_code') or e.get('error_tag') or e.get('tag') or e.get('code')
                    if code not in CODES:
                        match=re.search(r'\b(B[1-4]|R[1-3]|N[1-3]|S[12]|P1|T1)(?:_|\b)',str(e.get('status','')))
                        code=match.group(1) if match else None
                    if code not in CODES:continue
                    item={'event_id':e.get('event_id') or hashlib.sha256((str(file)+':'+str(line)).encode()).hexdigest()[:20],
                          'benchmark_id':b,'implementation_model':model,'taxonomy_code':code,
                          'phase':e.get('phase','unspecified'),'origin_model':e.get('origin_model') or ('shared_infrastructure' if str(e.get('phase','')).startswith('shared_') else model),'is_failure':code not in ['N3','P1'] and e.get('is_failure',e.get('failure',True)) is not False and not e.get('diagnostic_only',False) and e.get('in_scope',True) is not False,
                          'evidence_path':str(file.relative_to(ROOT)),'evidence_line':line,
                          'raw_event':e,'classification_basis':'explicit event code or explicit coded status; no count imputation'}
                    item=apply_event_review(item,event_reviews)
                    events.append(item);all_events.append(item)
            if model=='cuda_jl' and not evidence and not explicit.get('status') and any(e['taxonomy_code']=='S1' for e in events):
                status='reference_blocked'
            config_evidence=[]
            for cid,(d,c,p) in evidence:
                m=d['models'][model]
                model_traits=traits.get('implementation_traits',{}).get(model,{}) or {}
                config_traits=model_traits.get('configuration_traits',{}).get(cid,{})
                shared_config_traits=traits.get('configuration_overrides',{}).get(cid,{})
                class_map=shared_config_traits.get('implementation_class',{})
                config_class=config_traits.get('implementation_class') or (class_map.get({'cuda_jl':'julia','ka':'julia-ka','jacc':'julia-jacc'}[model]) if isinstance(class_map,dict) else class_map)
                scoped_effort_paths=[log_root/(b+'-julia-'+alias)/'configurations'/cid/'manifest.json'
                                     for alias in aliases
                                     if (log_root/(b+'-julia-'+alias)/'configurations'/cid/'manifest.json').is_file()]
                config_evidence.append({'configuration_id':cid,'run_id':d['run_id'],'result':str(p.relative_to(ROOT)),
                                        'configuration_role':disp.get(cid,{}).get('role',c.get('configuration_role','primary')),
                                        'numerical_agreement_kind':c.get('numerical_agreement_kind','declared_numeric_artifact_contract'),
                                        'verification_contract':c.get('verification_contract'),
                                        'numerical_agreement_claim':c.get('numerical_agreement_kind') not in ('no_numerical_outputs_execution_only','property_equivalence_not_identical_raw_values'),'raw_output_identity_claim':False if c.get('numerical_agreement_kind') in ('property_equivalence_not_identical_raw_values','no_numerical_outputs_execution_only') else None,
                                        'scoped_effort_records':[{'path':str(q.relative_to(ROOT)),'sha256':digest(q),'record':read(q),'scope':'Named configuration phase only; historical top-level effort is separate.'} for q in scoped_effort_paths],
                                        'timing_region_metadata_adjudication':d.get('timing_region_metadata_adjudication'),
                                        'native_reference_repaired':bool(c.get('native_reference_repaired')),
                                        'verification_contract_revised':bool(c.get('verification_contract_revised')),
                                        'native_behavior_known_scientific_defect':bool(c.get('native_behavior_known_scientific_defect')),
                                        'gpu_visibility':str(c.get('gpu',6)),
                                        'hardware_stratum':'single_v100' if str(c.get('gpu',6))=='6' else 'multiple_or_other_devices',
                                        'untouched_single_gpu_aggregate_eligible':c.get('numerical_agreement_kind')!='no_numerical_outputs_execution_only' and str(c.get('gpu',6))=='6' and not c.get('native_reference_repaired') and not c.get('verification_contract_revised') and not c.get('exclude_from_aggregate',False) and not c.get('exclude_untouched_native_aggregates',False) and not c.get('exclude_from_scientific_fidelity_aggregates',False) and not c.get('native_behavior_known_scientific_defect',False),
                                        'input_contract_metadata_erratum':c.get('input_contract_metadata_erratum'),'reported_metric_definition_erratum':c.get('reported_metric_definition_erratum'),'source_scope_adjudication':c.get('source_scope_adjudication'),'reference_provenance':c.get('reference_provenance'),'configuration_classification_adjudication':c.get('configuration_classification_adjudication'),
                                        'result_sha256':digest(p),'source_hash':m.get('source_hash'),
                                        'declared_environment':c.get('environment',c.get('toolchain')),
                                        'hardware_snapshot_path':str(p.with_name('hardware.txt').relative_to(ROOT)) if p.with_name('hardware.txt').exists() else None,
                                        'trials_correct':[t.get('correct') for t in m.get('trials',[])],
                                        'trials_seconds':[t.get('time_seconds') for t in m.get('trials',[])],
                                        'mean_last_four_seconds':m.get('mean_seconds'),'contract':c.get('input_contract'),
                                        'configuration_implementation_class':config_class,
                                        'implementation_ontology_override':config_traits or None,
                                        'shared_configuration_traits_override':shared_config_traits or None,
                                        'configuration_ontology_semantics':'Explicit configuration-level source review overrides benchmark-wide descriptive labels; null means no separate override recorded.',
                                        'numerical_evidence_strength':c.get('numerical_evidence_strength') or ('execution_only_no_numerical_outputs' if c.get('numerical_agreement_kind')=='no_numerical_outputs_execution_only' else 'property_score_artifacts_with_raw_structured_state_records' if c.get('numerical_agreement_kind')=='property_equivalence_not_identical_raw_values' else 'aggregate_count_and_error_sum_only' if b=='feynman-kac' else 'full_output_artifacts')})
            per_model=traits.get('implementation_traits',{}).get(model)
            record={'schema_version':2,'study':f'julia_abstractions_{population}_v2','benchmark_id':b,'implementation_model':model,
                    'status':status,'scientific_intent_record':f'../../domains/{b}.json',
                    'environment':{'julia_release':'1.12.6','cuda_jl':'6.2.1','kernel_abstractions':'0.9.45','jacc':'1.4.0',
                                   'scope':'Default study package environment only. Per-configuration declared_environment and toolchain commands override native-toolkit defaults; verify per-run snapshots before release comparisons.',
                                   'gpu':'Tesla V100-PCIE-32GB sm70','release_comparison_performed':False},
                    'implementation_ontology':per_model,
                    'implementation_ontology_normalized':normalize_traits(per_model),
                    'implementation_review_status':('per_model_source_review_recorded' if scan else 'status_review_recorded_no_implementation_source') if per_model else 'not_yet_reviewed_per_model',
                    'shared_benchmark_traits':traits if traits else None,
                    'shared_traits_caveat':'Legacy shared traits do not establish each implementation mapping or measured bottleneck.',
                    'source_syntax_evidence':scan,'configurations':config_evidence,
                    'failure_taxonomy':{'events':events,'observed_failure_event_count':sum(e['is_failure'] for e in events),
                                        'history_completeness':'partial' if (log/'compiler_attempts.jsonl').exists() or events else 'unknown',
                                        'missing_history_is_zero':False},
                    'effort':normalized_effort,'effort_field_provenance':effort_field_provenance,
                    'effort_manifest_provenance':{'selected_path':str(selected_manifest.relative_to(ROOT)),'selected_sha256':digest(selected_manifest) if selected_manifest.is_file() else None,'candidate_alias_records':[{'path':str(q.relative_to(ROOT)),'sha256':digest(q)} for q in manifest_candidates],'selection_rule':'Prefer source_sha256 matching current main, then latest mtime; aliases preserved separately, historical records not merged.'},
                    'source_commit':manifest.get('source_commit') or manifest.get('commit'),'source_commit_field':('source_commit' if manifest.get('source_commit') else 'legacy_manifest_commit' if manifest.get('commit') else None),'source_commit_scope':'Recorded implementation commit only; null is unknown/unjoined, not evidence of uncommitted code. Actual executed dependency hashes are separate.','explicit_outcome_evidence':explicit or None}
            # Some reviewed annotations store separate model classes in the
            # parent dictionary. Retain that explicit model entry, never infer
            # a model class from a shared string or source syntax.
            class_by_model=traits.get('implementation_class',{})
            if isinstance(class_by_model,dict):
                model_aliases=[model,{'cuda_jl':'julia','ka':'julia-ka','jacc':'julia-jacc'}[model],suffix]
                class_key=next((k for k in model_aliases if class_by_model.get(k) is not None),None)
                if class_key is not None:
                    record['declared_model_implementation_class']={
                        'value':class_by_model[class_key],
                        'annotation_path':str((STATE/'traits'/(b+'.json')).relative_to(ROOT)),
                        'annotation_sha256':digest(STATE/'traits'/(b+'.json')),
                        'field':'implementation_class.'+class_key,
                        'scope':'Explicit model-indexed source review, not a syntax-inferred class; configuration overrides take precedence.'}
                    if record['implementation_ontology_normalized'] is not None and record['implementation_ontology_normalized'].get('implementation_class') is None:
                        record['implementation_ontology_normalized']['implementation_class']=class_by_model[class_key]
            record['effort']['observed_harness_activity_windows']=usage_windows(b,model)
            scope_review=STATE/'usage_boundaries_300'/b/'measurement_scope_review.json'
            if scope_review.exists():record['effort']['activity_window_scope_review']={'path':str(scope_review.relative_to(ROOT)),'sha256':digest(scope_review),'record':read(scope_review)}
            profiles=read(STATE/'profile_summaries_300'/(b+'.json'),{})
            if profiles.get('models',{}).get(model):record['measured_profile_evidence']=profiles['models'][model]
            # Reference-only records can precede a manifest and live under any
            # established CUDA.jl log alias. Preserve separate phases verbatim;
            # do not lose them merely because manifest selection fell back to cuda.
            current_refs=[log_root/(b+'-julia-'+alias)/'current_reference_effort_300.json'
                          for alias in aliases
                          if (log_root/(b+'-julia-'+alias)/'current_reference_effort_300.json').is_file()]
            if current_refs:
                record['current_reference_repair_effort_records']=[
                    {'path':str(q.relative_to(ROOT)),'sha256':digest(q),'record':read(q),
                     'scope':'Current repair phase only; does not replace unknown historical original translation effort.'}
                    for q in current_refs]
                preferred=next((x for x in record['current_reference_repair_effort_records']
                                if x['path']==str((log/'current_reference_effort_300.json').relative_to(ROOT))),
                               record['current_reference_repair_effort_records'][0])
                record['current_reference_repair_effort']=preferred
            write(OUT/'implementations'/b/(model+'.json'),record)
            statuses.append({'benchmark_id':b,'model':model,'status':status,'ontology_review':record['implementation_review_status'],
                             'scientific_review':intent.get('review_status','unspecified'),
                             'primary_family':intent.get('primary_family',intent.get('primary_domain','unclassified')),
                             'verification_contract_revised':any(c.get('verification_contract_revised') for c in config_evidence),
                             'numerical_agreement_kinds':sorted({c.get('numerical_agreement_kind','declared_numeric_artifact_contract') for c in config_evidence})})
    # A reference barrier copied into several target ledgers is one origin event.
    def origin_fingerprint(event):
        raw=event['raw_event']
        return (event['benchmark_id'],event['origin_model'],event['taxonomy_code'],event['phase'],
                raw.get('configuration_id'),raw.get('root_cause',raw.get('cause',raw.get('reason',raw.get('symptom')))),
                raw.get('check_id',raw.get('compiler_check')),raw.get('source_candidate_id',raw.get('candidate')))
    # Direct native/shared events have no Julia target row; retain them in origin accounting.
    standalone=[]
    for item in [*all_reference_events,*all_infrastructure_events]:
        raw=item['raw_event'];code=raw.get('taxonomy_code') or raw.get('error_tag') or raw.get('tag') or raw.get('code')
        if code not in CODES:continue
        model='cuda' if raw.get('implementation_model') in ['cuda','native_cuda'] else 'shared_infrastructure'
        standalone.append({**item,'event_id':raw.get('event_id') or hashlib.sha256((item['evidence_path']+':'+str(item['evidence_line'])).encode()).hexdigest()[:20],
            'implementation_model':model,'origin_model':raw.get('origin_model',model),'taxonomy_code':code,'phase':raw.get('phase','unspecified'),
            'is_failure':code not in ['N3','P1'] and raw.get('is_failure',raw.get('failure',True)) is not False and not raw.get('diagnostic_only',False) and raw.get('in_scope',True) is not False,
            'classification_basis':'Explicit standalone native/shared event; no target failure imputation'})
    standalone=[apply_event_review(e,event_reviews) for e in standalone]
    origin_population=[*all_events,*standalone]
    copied_origin_groups={origin_fingerprint(e) for e in origin_population if e['origin_model']!=e['implementation_model']}
    origin_events={}
    for event in origin_population:
        fp=origin_fingerprint(event)
        key=('reviewed_observation',event['canonical_observation_id'],event['taxonomy_code']) if event.get('canonical_observation_id') else ('copied_origin',fp) if fp in copied_origin_groups else ('event_id',event['event_id'])
        origin_events.setdefault(key,event)
    summary={'generated_at':now,'target_benchmarks':population,'implementation_slots':3*population,
             'status_counts':{m:dict(Counter(r['status'] for r in statuses if r['model']==m)) for m in MODELS},
             'per_model_review_counts':{m:dict(Counter(r['ontology_review'] for r in statuses if r['model']==m)) for m in MODELS},
             'scientific_review_counts':dict(Counter(d['scientific_intent'].get('review_status','unspecified') for d in domain_rows)),
             'provisional_family_counts':dict(Counter(d['scientific_intent'].get('primary_family','unclassified') for d in domain_rows)),
             'scientific_family_counts':dict(Counter(d['scientific_intent'].get('primary_family','unclassified') for d in domain_rows)),
             'legacy_summary_aliases':{'provisional_family_counts':'Compatibility alias for scientific_family_counts; review_status fields determine actual review depth.'},
             'observed_taxonomy_events':{m:dict(Counter(e['taxonomy_code'] for e in all_events if e['implementation_model']==m)) for m in MODELS},
             'observed_origin_taxonomy_events':{m:dict(Counter(e['taxonomy_code'] for e in origin_events.values() if e['origin_model']==m)) for m in [*MODELS,'cuda','shared_infrastructure']},
             'origin_event_semantics':'Explicit origin_model overrides affected target; shared_* phases grouped as shared infrastructure. Exact source-line reviews group duplicate observations/correct origins; raw failure:false is honored. Otherwise copied reference notifications use benchmark/origin/code/phase/cause/check/candidate. Process effort remains separate; partial histories, not total rates.',
             'recorded_shared_infrastructure_events':len(all_infrastructure_events),'recorded_native_reference_events':len(all_reference_events),'failure_history_totals_known':False,'all_names_ordinary_verified':all(sum(1 for r in statuses if r['model']==m and r['status']=='verified_named_configuration')==population for m in MODELS),
             'meaning':'Complete inventory denominator; current outcomes, unknown records and partial histories stay explicit.'}
    write(OUT/'index.json',{'schema_version':2,'summary':summary,'records':statuses})
    write(OUT/'failure_events.json',{'schema_version':2,'events':all_events,'native_reference_events':all_reference_events,'shared_infrastructure_events':all_infrastructure_events,'origin_events':list(origin_events.values()),'count_interpretation':'Observed event rows by model, not all historical failures. N3/P1 excluded from failed-event totals.'})
    write(STATE/f'study_{population}_status.json',{'summary':summary,'records':statuses})
    write(PAPER/f'research/data/study_{population}_status.json',{'summary':summary,'records':statuses})
    write(PAPER/f'research/data/scientific_domains_{population}.json',domain_rows)
    write(PAPER/f'research/data/failure_events_{population}.json',{'events':all_events,'native_reference_events':all_reference_events,'shared_infrastructure_events':all_infrastructure_events,'origin_events':list(origin_events.values()),'partial_history':True})
    print(json.dumps(summary))

if __name__=='__main__':main()
