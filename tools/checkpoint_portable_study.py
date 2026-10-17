#!/usr/bin/env python3
"""Write durable combined checkpoint from local study evidence and both git heads."""
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];state=root/'.porting-state';paper=root/'2026-PASC-Hecbench-Julia-paper'
def git(path,*args):return subprocess.check_output(['git',*args],cwd=path,text=True).strip()
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%S%fZ');out=paper/'research/checkpoints'/('checkpoint_'+stamp+'.md')
cohort=json.loads((state/'portable_julia/batch_100.json').read_text());results={};configurations={}
for f in sorted((state/'portable_julia/runs').glob('*/result.json')):
 d=json.loads(f.read_text())
 if any(d.get('variant_verified',{}).values()):configurations[(d['benchmark'],d.get('configuration_id') or json.loads(f.with_name('config.json').read_text())['configuration_id'])]={'run_id':d['run_id'],'variants':d['variant_verified'],'matched4':d.get('matched_performance_valid'),'configuration':d.get('configuration_id'),'models':{k:{'trials_seconds':[t.get('time_seconds') for t in m.get('trials',[])],'mean_seconds':m.get('mean_seconds'),'source_hash':m.get('source_hash')} for k,m in d['models'].items()},'references_agree':d.get('reference_agreement'),'local_evidence':str(f.relative_to(root)),'comparisons':d.get('comparisons')}
for (benchmark,configuration),record in configurations.items():
 record['configuration']=configuration
 if benchmark not in results or record['matched4'] or not results[benchmark]['matched4']:results[benchmark]=record
heads={'source':git(root,'rev-parse','HEAD'),'paper':git(paper,'rev-parse','HEAD')};dirty={k:git(p,'status','--short') for k,p in [('source',root),('paper',paper)]}
instructions={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['AGENT_INSTRUCTIONS_CODEX_Julia.md','JULIA_CORRECTNESS_CHECK.md','paper_writing.md']}
pairs=sum(x['matched4'] for x in results.values());payload={'created_at':now.isoformat(),'heads':heads,'instruction_hashes':instructions,'state':'.porting-state','frozen_baseline':'portable_julia/baseline_300.json','cohort':cohort,'latest_evidence':results,'all_configuration_evidence':list(configurations.values()),'dirty_files':dirty}
out.write_text('# Active translation and paper checkpoint\n\n'+f'{pairs}/100 matched benchmark pairs in latest local bundles. Work remains mandatory.\n\n'+'''## Resume instructions
Read AGENT_INSTRUCTIONS_CODEX_Julia.md, JULIA_CORRECTNESS_CHECK.md, paper_writing.md, then this checkpoint. Preserve unrelated dirty work, especially deleted AGENT_INSTRUCTIONS_CODEX.md and md5hash caches. Workers have disjoint ownership below; root alone stages/commits. All GPU execution uses GPU6 and flock /tmp/hecbench-julia-study-gpu.lock. Runner locks itself. Do not start overlapping GPU runs.

## Measurement and evidence
Five in-process trials per model, all correct, trial1 time excluded; mean lastfour. Kernel/JIT warmup before trial1; identical fixtures and regions for bothreferences and bothvariants. Reuse unchanged validated reference/KA trials with hash validation rather than additional series. Numerical and timing invalidations are retained, never silently deleted. Native originalkernels unchanged; named studyfixtures and algorithm/layout adaptations explicit. Raw fullbinary artifacts and SQLiteDB remain localSTATE and are not in git; paper/research/data contains portable result/config/hash/trial exports. Original 300-pass historicalbaseline is frozen; sourceaudit87 is a screen, not87 established failures.

## Work status and next actions
Per-worker selected lists appear in cohort. Continue each assigned benchmark KA first, then JACC, followed by logs/manifests, paperobservation, and separate variantcommits. See live collaborationagent messages and local manifests for current in-flight runs; discover newest run without result.json and tail its model logs. Root current adapter work can be found via uncommitted src/*-julia/study_config*.json. New assignments after worker slices remain necessary until100 matched pairs. Reference repair is authorized where straightforward and must preserve originalsource, separate correctionphase cost, and perform full numericalverification. A deferred/unsupportedcase does not countas verified pair; maintain rejection/refillhistory.

Rootnext: commit any readyvariants after source/evidencereview, export paperdata with research/analysis/export_verified_study.py --source-root <root>, regenerate charts with plot_verified_study.py (matplotlib isolated /tmp/pasc-julia-plot-tools), compile make inpaperrepo, updateguidecounts and write newcheckpoint. All remaining100cohortports, profiling/ontologyanalysis, and finalpaperconclusions remain mandatory.

## Cost and compiler-loop records
Per-variant records STATE/porting_logs/<bench>-julia-{ka,jacc}/manifest.json, token_usage.jsonl, compiler_attempts.jsonl, context_checkpoint*.md. Some sources export translation_accounting.json. Exactpervariant modelusage unavailable: source/summarycharproxies remain partial; unknowncounts null. ParentCodexsession exactcumulativeusage snapshots STATE/portable_julia/session_usage are mixedconcurrentwork and remain shared/unallocated. Do not add cachedtokens to inputtokens. Sourcecandidate count, compiler/JITchecks, feedbackrepairs and five timingtrials are separate. Existing CUDA.jl cost unchanged. Record new unsupportedfeatures with installedAPI evidence, notenvironmentfailures.

## Paper status
ACM sigconf manuscript sections00–09, commonfour-code example and combinedmethoddiagram present. Latest manuscript build and exact count are recorded in paper/research/data/coverage.json and build/main.log. RQ1 dualreference fulloutputs and accepteddrift recorded; RQ2 phase-specific attempts and partialtokenproxies; RQ3 matchedfourmodeldata; RQ4 reverse profiles plus usefulFP32 triadroofline and controlled JACC49152B→0B resourceintervention (profile161→64us, sharedoccupancylimit2→32). Other motifbottlenecks remain hypotheses. Paperresearch/data/configuration_dispositions.json retains oldresource defaults asablation, notduplicateprimaryconfigurations. Rootresource-tuning batch script /tmp/backfill_jacc_resource_policy.py, shellsession3017; preservebaselines and commitnewversionsseparately. RQ5 conditionalguidance only, no universalwinner. Sourceontology labels require manualreview, native/Julia mappings and namedfixture deviations explicitlyconsidered. Scope100baselinebenchmarks is feasibility-selected, notrandomrepresentativesample; namedsubkernel cases do not imply entireCLI/precisioncoverage.

## Captured state
```json
'''+json.dumps(payload,indent=2)+'\n```\n')
latest={'checkpoint_id':stamp,'path':str(out.relative_to(root)),'created_at':now.isoformat(),'status':'active_100_pair_translation','verified_pairs':pairs,'target_pairs':100,'performance_protocol':'five_trials_last_four_v1','source_head':heads['source'],'paper_head':heads['paper']};tmp=out.parent/'LATEST.json.tmp';tmp.write_text(json.dumps(latest,indent=2)+'\n');tmp.replace(out.parent/'LATEST.json');print(json.dumps(latest))
