#!/usr/bin/env python3
"""Export the full census without converting unknowns into successes or failures."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, re, subprocess

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.porting-state/portable_julia'
PAPER=ROOT/'2026-PASC-Hecbench-Julia-paper'
OUT=ROOT/'ontology/studies/julia_abstractions_300'
MODELS={'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}
CODES=['B1','B2','B3','B4','R1','R2','R3','N1','N2','N3','S1','S2','P1','T1']

def read(p,default=None):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return default

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def jsonlines(p):
    if not p.exists():return []
    out=[]
    for i,line in enumerate(p.read_text().splitlines(),1):
        if not line.strip():continue
        try:out.append((i,json.loads(line)))
        except ValueError:continue # A worker may be appending; never infer a zero.
    return out

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
        'work_item_mapping':pick('work_item_mapping','mapping','algorithm_and_mapping'),
        'layout':pick('layout','data_layout'),
        'storage_precision':pick('storage_precision') or precision.get('storage'),
        'compute_precision':pick('compute_precision') or precision.get('compute'),
        'memory_hierarchy_and_access':pick('memory_hierarchy','memory_access','memory'),
        'synchronization':pick('synchronization'),
        'atomics_and_reductions':pick('atomic_operations','atomics','reductions'),
        'resource_policy':pick('resource_policy','resource_geometry'),
        'launch_geometry':pick('launch_geometry','resource_geometry'),
        'library_backend_boundary':pick('library_boundary','backend_boundary','backend_extensions','actual_backend'),
        'host_control_boundary':pick('host_control_boundary'),
        'fidelity_status_and_scope':pick('fidelity_scope','fidelity_status','status'),
        'source_review_evidence':pick('source_evidence','source_sha256','source_revision'),
        'null_interpretation':'Not recorded in this source review; never assume absent feature or zero resource use.'}

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    baseline=read(STATE/'baseline_300.json');cohort=read(STATE/'batch_300.json')
    selected=set(cohort['benchmarks']);base={x['bench']:x for x in baseline['benchmarks']}
    adjud=read(PAPER/'research/data/adjudications.json',{});disp=read(PAPER/'research/data/configuration_dispositions.json',{})
    latest={}
    for p in sorted((STATE/'runs').glob('*/result.json')):
        d=read(p,{})
        if d.get('benchmark') not in selected or adjud.get(d.get('run_id'),{}).get('exclude_from_performance'):continue
        c=read(p.with_name('config.json'),{});cid=c.get('configuration_id')
        if disp.get(cid,{}).get('role','primary') not in ['primary','corrected_reference_secondary']:continue
        for model in MODELS:
            good=(d.get('reference_agreement') and d.get('models',{}).get('cuda_jl',{}).get('complete')) if model=='cuda_jl' else d.get('variant_verified',{}).get(model)
            if good:latest[(d['benchmark'],model,cid)]=(d,c,p)
    domains=read(STATE/'scientific_domains_300.json',{})
    statuses=[];all_events=[];all_reference_events=[];domain_rows=[]
    for b in cohort['benchmarks']:
        traits=read(STATE/'traits'/(b+'.json'),{})
        intent=domains.get(b) or traits.get('scientific_intent')
        if not intent:intent={'primary_family':LOOKUP.get(b,'unclassified'), 'review_status':'provisional_category_requires_source_review','scientific_task':None,'community_use_case':None,'evidence':[],'classification_method':'Curated name/motif grouping to organize source review; not validated application provenance.'}
        domain={'schema_version':2,'benchmark_id':b,'scientific_intent':intent,
                'base_ontology':{'path':base[b].get('ontology_path'),'frozen_sha256':base[b].get('ontology_sha256'),'status':'inferred base; not measured evidence'},
                'historical_pass_method':base[b].get('method'),'historical_pass_is_strengthened_verification':False}
        write(OUT/'domains'/(b+'.json'),domain);domain_rows.append(domain)
        for line,event in jsonlines(STATE/'failure_events_300'/(b+'.jsonl')):
            if event.get('implementation_model') in ['cuda','native_cuda']:
                all_reference_events.append({'benchmark_id':b,'evidence_path':str((STATE/'failure_events_300'/(b+'.jsonl')).relative_to(ROOT)),'evidence_line':line,'raw_event':event})
        for model,suffix in MODELS.items():
            folder=ROOT/'src'/(b+'-julia');path=folder/(b+'-julia-'+suffix)/'main.jl'
            if model=='cuda_jl' and not path.exists():path=folder/'main.jl'
            log=ROOT/'.porting-state/porting_logs'/(b+'-julia-'+('ka' if model=='ka' else 'jacc' if model=='jacc' else 'cuda'))
            manifest=read(log/'manifest.json',{})
            evidence=[(cid,v) for (bench,m,cid),v in latest.items() if bench==b and m==model]
            scan=scan_source(path)
            status='verified_named_configuration' if evidence else 'in_progress' if manifest or (model!='cuda_jl' and path.exists()) else 'historical_pass_not_reaudited' if model=='cuda_jl' else 'not_started'
            explicit=read(STATE/'status_300'/b/(model+'.json'),{})
            if evidence and any(v[1].get('native_reference_repaired') for _,v in evidence):status='conditionally_verified_native_reference_repaired'
            if explicit.get('status') and not evidence:status=explicit['status']
            events=[]
            sources=[(log/'compiler_attempts.jsonl',False),(STATE/'failure_events_300'/(b+'.jsonl'),True)]
            normalized_links={(e.get('preserved_event_source'),e.get('check_id')) for _,e in jsonlines(STATE/'failure_events_300'/(b+'.jsonl')) if e.get('implementation_model',e.get('model'))==model and e.get('preserved_event_source') and e.get('check_id') is not None}
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
                          'phase':e.get('phase','unspecified'),'origin_model':e.get('origin_model') or ('shared_infrastructure' if str(e.get('phase','')).startswith('shared_') else model),'is_failure':code not in ['N3','P1'] and e.get('is_failure',True) is not False and not e.get('diagnostic_only',False) and e.get('in_scope',True) is not False,
                          'evidence_path':str(file.relative_to(ROOT)),'evidence_line':line,
                          'raw_event':e,'classification_basis':'explicit event code or explicit coded status; no count imputation'}
                    events.append(item);all_events.append(item)
            if model=='cuda_jl' and not evidence and not explicit.get('status') and any(e['taxonomy_code']=='S1' for e in events):
                status='reference_blocked'
            config_evidence=[]
            for cid,(d,c,p) in evidence:
                m=d['models'][model]
                config_evidence.append({'configuration_id':cid,'run_id':d['run_id'],'result':str(p.relative_to(ROOT)),
                                        'native_reference_repaired':bool(c.get('native_reference_repaired')),
                                        'gpu_visibility':str(c.get('gpu',6)),
                                        'hardware_stratum':'single_v100' if str(c.get('gpu',6))=='6' else 'multiple_or_other_devices',
                                        'untouched_single_gpu_aggregate_eligible':str(c.get('gpu',6))=='6' and not c.get('native_reference_repaired') and not c.get('exclude_from_aggregate',False) and not c.get('exclude_untouched_native_aggregates',False),
                                        'reference_provenance':c.get('reference_provenance'),
                                        'result_sha256':digest(p),'source_hash':m.get('source_hash'),
                                        'declared_environment':c.get('environment',c.get('toolchain')),
                                        'hardware_snapshot_path':str(p.with_name('hardware.txt').relative_to(ROOT)) if p.with_name('hardware.txt').exists() else None,
                                        'trials_correct':[t.get('correct') for t in m.get('trials',[])],
                                        'trials_seconds':[t.get('time_seconds') for t in m.get('trials',[])],
                                        'mean_last_four_seconds':m.get('mean_seconds'),'contract':c.get('input_contract'),
                                        'numerical_evidence_strength':c.get('numerical_evidence_strength','full_output_artifacts')})
            per_model=traits.get('implementation_traits',{}).get(model)
            record={'schema_version':2,'study':'julia_abstractions_300_v2','benchmark_id':b,'implementation_model':model,
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
                    'effort':{k:manifest.get(k) for k in ['candidate_attempts','compiler_checks','compiler_feedback_repairs','runtime_repairs','numerical_repairs','translation_tokens']},
                    'source_commit':manifest.get('source_commit'),'explicit_outcome_evidence':explicit or None}
            write(OUT/'implementations'/b/(model+'.json'),record)
            statuses.append({'benchmark_id':b,'model':model,'status':status,'ontology_review':record['implementation_review_status'],
                             'scientific_review':intent.get('review_status','unspecified'),
                             'primary_family':intent.get('primary_family',intent.get('primary_domain','unclassified'))})
    # A reference barrier copied into several target ledgers is one origin event.
    def origin_fingerprint(event):
        raw=event['raw_event']
        return (event['benchmark_id'],event['origin_model'],event['taxonomy_code'],event['phase'],
                raw.get('configuration_id'),raw.get('root_cause',raw.get('cause',raw.get('reason',raw.get('symptom')))),
                raw.get('check_id',raw.get('compiler_check')),raw.get('source_candidate_id',raw.get('candidate')))
    copied_origin_groups={origin_fingerprint(e) for e in all_events if e['origin_model']!=e['implementation_model']}
    origin_events={}
    for event in all_events:
        fp=origin_fingerprint(event)
        key=('copied_origin',fp) if fp in copied_origin_groups else ('event_id',event['event_id'])
        origin_events.setdefault(key,event)
    summary={'generated_at':now,'target_benchmarks':300,'implementation_slots':900,
             'status_counts':{m:dict(Counter(r['status'] for r in statuses if r['model']==m)) for m in MODELS},
             'per_model_review_counts':{m:dict(Counter(r['ontology_review'] for r in statuses if r['model']==m)) for m in MODELS},
             'scientific_review_counts':dict(Counter(d['scientific_intent'].get('review_status','unspecified') for d in domain_rows)),
             'provisional_family_counts':dict(Counter(d['scientific_intent'].get('primary_family','unclassified') for d in domain_rows)),
             'observed_taxonomy_events':{m:dict(Counter(e['taxonomy_code'] for e in all_events if e['implementation_model']==m)) for m in MODELS},
             'observed_origin_taxonomy_events':{m:dict(Counter(e['taxonomy_code'] for e in origin_events.values() if e['origin_model']==m)) for m in [*MODELS,'cuda','shared_infrastructure']},
             'origin_event_semantics':'Explicit origin_model overrides affected target; shared_* phases grouped as shared infrastructure. Repeated reference notifications deduplicated by benchmark/origin/code/phase/cause/check/candidate. Partial histories, not total rates.',
             'recorded_native_reference_events':len(all_reference_events),'failure_history_totals_known':False,'all_300_verified':all(sum(1 for r in statuses if r['model']==m and r['status']=='verified_named_configuration')==300 for m in MODELS),
             'meaning':'Complete inventory denominator; current outcomes, unknown records and partial histories stay explicit.'}
    write(OUT/'index.json',{'schema_version':2,'summary':summary,'records':statuses})
    write(OUT/'failure_events.json',{'schema_version':2,'events':all_events,'native_reference_events':all_reference_events,'count_interpretation':'Observed event rows by model, not all historical failures. N3/P1 excluded from failed-event totals.'})
    write(STATE/'study_300_status.json',{'summary':summary,'records':statuses})
    write(PAPER/'research/data/study_300_status.json',{'summary':summary,'records':statuses})
    write(PAPER/'research/data/scientific_domains_300.json',domain_rows)
    write(PAPER/'research/data/failure_events_300.json',{'events':all_events,'native_reference_events':all_reference_events,'partial_history':True})
    print(json.dumps(summary))

if __name__=='__main__':main()
