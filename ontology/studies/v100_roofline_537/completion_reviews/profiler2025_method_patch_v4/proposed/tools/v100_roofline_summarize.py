#!/usr/bin/env python3
"""Freeze completed raw profiles and regenerate auditable kernel-level exports.

Only finished collection attempts are imported. The model table includes every
assessment slot; sample points never become whole-benchmark bottleneck claims.
"""
import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import json
import re
import copy
import fcntl
from v100_profiler_method import LEGACY_VERSION
from pathlib import Path
from v100_roofline_allreduce_admission import METHOD as ALLREDUCE_METHOD, HOLD as ALLREDUCE_HOLD, read_raw_slot as allreduce_files, validate as validate_allreduce_method


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def freeze(slot, target, root):
    manifest = []
    target.mkdir(parents=True, exist_ok=True)
    for name in ('plan.json','profile_provenance.json','counters.csv','derived.json',
                 'execution.log','build.log','profiling_auxiliary_manifest.json',
                 'collector_source.py','parser_source.py','write_guard_source.py',
                 'plan_diff.json','build_manifest.json','compiler_manifest.json',
                 'plan_before_interpreters.json','interpreter_plan_diff.json',
                 'build_outcome.json','counter_snapshot_at_derivation.csv',
                 'counter_snapshot_recovery.json','tensor_mode_loader_source.py',
                 'tensor_mode_decisions.json','parent_profile_provenance.json',
                 'parent_parser_source.py','half_calibration_snapshot.json',
                 'recovery_tool_source.py','counter_derivation_recovery.json',
                 'completion_source_proof.json','patches.json',
                 'plan_before_replay_review.json','replay_metadata_derivation.json',
                 'assert_child_diagnostics.json','remaining_infra_source_proof.json',
                 'egs_build_manifest.json','source_proof.json','application_replay_review.json',
                 'application_replay_pass_artifacts.json','profile_window_review.json','isolated_assertion_review.json',
                 'negative_prechecks.json','positive_profile_outcome.json','joined_execution_contract.json',
                 'profiler_method_review.json','profiler_method_bundle.json','profiler_method_validator.py',
                 'profiler_method_preflight.json','profiler_version.log','profiler_version_payload.bin',
                 'counter_quiescence_snapshot.json','profiler_method_post_profile_check.json',
                 'isolated_assertion_negative_raw.json','phase_spec.json',
                 'pre_profile_checks.log','post_profile_checks.log',
                 'adapter_source.py','allreduce_two_device_review.json',
                 'allreduce_cpu_review.json','allreduce_admission_source.py',
                 'allreduce_execution_authorization.json'):
        source = slot/name
        if not source.is_file():
            continue
        data = source.read_bytes()
        dest = target/(name+'.gz')
        if dest.exists():
            if gzip.decompress(dest.read_bytes()) != data:
                raise RuntimeError('Refuse mutation of frozen profile: '+str(dest))
        else:
            dest.write_bytes(gzip.compress(data, mtime=0))
        manifest.append({'original_name':name,'uncompressed_sha256':sha(data),
            'uncompressed_bytes':len(data),'compressed_sha256':sha(dest.read_bytes()),
            'compressed_bytes':dest.stat().st_size,
            'path':str(dest.relative_to(root))})
    dump(target/'manifest.json', {'schema_version':1,'files':manifest,
        'retention':'Raw counters, logs, plans and provenance retained losslessly. Disposable new scratch auxiliary outputs are represented only by a manifest; no new correctness certification is claimed.'})
    return str((target/'manifest.json').relative_to(root))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,required=True)
    a=ap.parse_args();root=a.source_root.resolve()
    study=root/'ontology/studies/v100_roofline_537'
    quality_path=study/'profile_quality_reviews.json'
    quality_reviews=json.loads(quality_path.read_text()).get('profiles',{}) if quality_path.exists() else {}
    quality_sha=sha(quality_path.read_bytes()) if quality_path.exists() else None
    applicability_path=study/'completion_reviews/infra_am/runtime_no_kernel_applicability.json'
    applicability_doc=json.loads(applicability_path.read_text())if applicability_path.exists()else{}
    applicability={(row['benchmark'],row['model']):row for row in applicability_doc.get('rows',[])}
    applicability_sha=sha(applicability_path.read_bytes())if applicability_path.exists()else None
    reviews={};review_hashes={}
    review_path=study/'ROI_REVIEW.json.gz'
    if review_path.exists():
        receipt=json.loads((study/'ROI_REVIEW_RECEIPT.json').read_text())
        packed=review_path.read_bytes();unpacked=gzip.decompress(packed)
        if sha(packed)!=receipt['sha256'] or sha(unpacked)!=receipt['uncompressed_sha256']:
            raise RuntimeError('ROI review hash mismatch')
        reviews={(x['benchmark'],x['model']):x for x in json.loads(unpacked)['records']}
        review_hashes.update({key:receipt['uncompressed_sha256'] for key in reviews})
    for candidate_path in sorted((study/'roi_reviews').glob('*.json.gz')):
        candidate_raw=gzip.decompress(candidate_path.read_bytes())
        candidate_doc=json.loads(candidate_raw)
        for entry in candidate_doc['records']:
            entry['_candidate_review_only']=True
            key=(entry['benchmark'],entry['model'])
            if key in reviews:raise RuntimeError('Overlapping ROI review authority: '+str(key))
            reviews[key]=entry;review_hashes[key]=sha(candidate_raw)
    imported=[];best_scores={}
    for provpath in sorted(a.raw_root.glob('batch*/*/*/profile_provenance.json')):
        prov=json.loads(provpath.read_text())
        if not prov.get('finished_at'):
            continue
        method_error = None
        if prov.get('status')=='not_started_profiler_method_preflight_hold':
            method_error='Retained preflight rejection; no target/GPU/build execution'
        elif prov.get('profiler_method') or (prov.get('profiler_version') and prov['profiler_version']!=LEGACY_VERSION):
            try:
                import v100_profiler_method as method_gate
                names=('profiler_method_review.json','profiler_method_bundle.json','profiler_method_validator.py',
                       'profiler_version.log','profiler_version_payload.bin','profiler_method_preflight.json','derived.json',
                       'counters.csv','counter_quiescence_snapshot.json','profiler_method_post_profile_check.json')
                available={n:(provpath.parent/n).read_bytes() for n in names if (provpath.parent/n).is_file()}
                method_gate.audit_profile(prov,available)
            except Exception as error:
                method_error=repr(error)
        slot=provpath.parent;plan=json.loads((slot/'plan.json').read_text())
        benchmark=plan['benchmark'];model=plan['model'];batch=slot.parent.parent.name
        target=study/'profiles'/benchmark/model/batch
        quality=quality_reviews.get(str(target.relative_to(root)))
        if quality and quality['raw_provenance_sha256']!=sha(provpath.read_bytes()):
            raise RuntimeError('Profile quality review provenance mismatch')
        effective_status=('profile_completed_quality_unverified' if quality and prov['status']=='profile_process_completed' else prov['status'])
        method_admission=None;method_hold=False
        if prov.get('allreduce_two_device_method'):
            try:
                method_admission=validate_allreduce_method(allreduce_files(slot),root)
            except Exception as error:
                method_hold=True
                effective_status=ALLREDUCE_HOLD
                method_admission={'status':'validation_hold','error':str(error),
                    'method_id':prov.get('allreduce_two_device_method',{}).get('method_id'),
                    'positions_publishable':False,'underlying_process_status':prov['status']}
        runtime_review=applicability.get((benchmark,model))
        runtime_applicable=(runtime_review is not None and runtime_review['parent_profile']['sha256']==sha(provpath.read_bytes())
            and prov['status']=='profile_completed_counter_derivation_failed'
            and prov.get('profile',{}).get('returncode')==0 and prov.get('profile',{}).get('stop_reason')is None)
        if runtime_applicable:effective_status='roofline_not_applicable_no_device_kernels'
        manifest=freeze(slot,target,root)
        contract=study/'selected_contracts'/benchmark
        contract.mkdir(parents=True,exist_ok=True)
        for source_name,expected in [('result.json',plan['reference_result_sha256']),
                                     ('config.json',plan['raw_config_sha256'])]:
            source=Path(plan['reference_result']).with_name(source_name)
            source_bytes=source.read_bytes()
            if sha(source_bytes)!=expected:
                raise RuntimeError('Selected contract bytes changed: '+str(source))
            destination=contract/(source_name+'.gz')
            if destination.exists() and gzip.decompress(destination.read_bytes())!=source_bytes:
                raise RuntimeError('Refuse mutation of frozen selected contract')
            if not destination.exists():destination.write_bytes(gzip.compress(source_bytes,mtime=0))
        path=study/'benchmarks'/(benchmark+'.json');ledger=json.loads(path.read_text())
        state=ledger['models'][model]
        attempts=state.setdefault('profile_attempts',[])
        attempt={'profile_id':str(target.relative_to(root)),'status':effective_status if prov.get('allreduce_two_device_method') else prov['status'],
                 'recorded_status':prov['status'],'method_admission':method_admission,
                 'finished_at':prov['finished_at'],'evidence_manifest':manifest,
                 'collector_sha256':prov['collector_sha256'],
                 'counters_sha256':prov.get('counters_sha256'),
                 'selector_scope':prov['selector_scope'],
                 'raw_process_status':prov['status'],
                 'profiler_method':prov.get('profiler_method'),
                 'profiler_version':prov.get('profiler_version')}
        if method_error:
            attempt.update(status='profile_method_unapproved',method_validation_error=method_error)
        previous=next((x for x in attempts if x['profile_id']==attempt['profile_id']),None)
        if previous is None:
            attempts.append(attempt)
        elif prov.get('allreduce_two_device_method'):
            # Revalidation is fresh metadata, not a new process/capture attempt.
            previous.update(status=effective_status,recorded_status=prov['status'],method_admission=method_admission)
        if attempt['finished_at']>=state.get('latest_profile_attempt',{}).get('finished_at',''):
            state['latest_profile_attempt']=attempt
        if method_error:
            for saved_attempt in attempts:
                if saved_attempt['profile_id']==attempt['profile_id']:saved_attempt.update(attempt)
            if str(target.relative_to(root)) in state.get('profile_ids',[]):
                state.update(profiling_status='profile_method_unapproved',profile_ids=[],sampled_kernel_count=0,sampled_kernel_precision_regimes={},sampled_kernel_predicted_resource_limits={},roi_coverage='method_unapproved_no_positioning',roofline_position=None,roofline_region=None,observed_bottleneck=None)
                state.pop('kernel_observations',None)
            dump(path,ledger)
            continue
        observation=slot/'derived.json'
        records=json.loads(observation.read_text())['records'] if observation.exists() and not method_hold else []
        review=reviews.get((benchmark,model))
        annotated_selected = False
        if review and observation.is_file() and not method_hold and review['configuration_id']==plan['configuration_id']:
            if review['reference_result']['sha256']!=plan['reference_result_sha256']:
                raise RuntimeError('ROI selected result mismatch')
            annotated=copy.deepcopy(records)
            exact={x['profile_identity']['Kernel Name']+'|'+x['profile_identity']['Grid Size']+'|'+x['profile_identity']['Block Size']:x
                   for x in review['kernel_mappings']}
            same_capture=(review.get('derived',{}).get('sha256')==sha(observation.read_bytes())) if observation.exists() else False
            for kernel in annotated:
                identity=kernel['identity'];key=identity['Kernel Name']+'|'+identity['Grid Size']+'|'+identity['Block Size']
                mapping=exact.get(key) if same_capture else None
                if mapping:
                    role={'stage_candidates':mapping['semantic_stage_candidates'],
                          'semantic_family_in_primary_ROI':mapping['semantic_family_in_primary_ROI'],
                          'sample_phase':mapping['sample_phase'],
                          'applicability':mapping['applicability'],
                          'mapping_status':mapping['mapping_status']}
                else:
                    matched=[x for x in review['stages'] if x.get('model_kernel_regex')
                             and re.search(x['model_kernel_regex'],identity['Kernel Name'])]
                    role={'stage_candidates':[x['stage_id']for x in matched],
                          'semantic_family_in_primary_ROI':(matched[0].get('semantic_family_in_primary_ROI')
                              if len(matched)==1 and 'semantic_family_in_primary_ROI' in matched[0]
                              else True if len(matched)==1 else None),
                          'sample_phase':'first_source_family_invocation_partial_not_complete_ROI',
                          'applicability':[x.get('applicability',review.get('applicability','unreviewed'))for x in matched],
                          'mapping_status':'source_family_match_new_capture_requires_case_review' if matched else 'unassigned'}
                    if review.get('_candidate_review_only'):
                        proved=(len(matched)==1 and matched[0].get('ROI_membership_proof')
                                and matched[0].get('ROI_membership_status')in[
                                    'source_timed_closure_inclusion_proved','source_untimed_audit_or_setup_exclusion_proved',
                                    'source_untimed_audit_exclusion_proved','source_setup_exclusion_proved'])
                        if not proved:role['semantic_family_in_primary_ROI']=None
                        role['mapping_status']=('source_closure_membership_proved_sample_phase_unproved' if proved
                                               else 'source_callsite_candidate_ROI_unresolved')
                if kernel['scope_context'].get('sample_in_primary_timed_invocation') is True and kernel['scope_context'].get('retained_trial'):
                    role['sample_phase']=kernel['scope_context']['sample_phase']
                kernel['ROI_annotation']=role
                stage_label=(';'.join(role['stage_candidates'])or'unassigned')
                if review.get('_candidate_review_only') and role['semantic_family_in_primary_ROI'] is None and stage_label!='unassigned':
                    stage_label='candidate:'+stage_label
                kernel['scope_context'].update(ROI_stage=stage_label,
                    semantic_family_in_primary_ROI=role['semantic_family_in_primary_ROI'],
                    sample_phase=role['sample_phase'],ROI_review_sha256=review_hashes[(benchmark,model)],
                    sample_in_primary_timed_invocation=(mapping.get('sample_in_primary_timed_invocation')
                                                       if mapping else kernel['scope_context'].get('sample_in_primary_timed_invocation')))
                if same_capture and review.get('supplemental_binding_warning'):
                    kernel['scope_context']['source_capture_binding_qualification']=review['supplemental_binding_warning']
            annotation_doc={'schema_version':1,'records':annotated,'kernel_count':len(annotated),
                'unannotated_derived_sha256':sha(observation.read_bytes()),
                'ROI_review_sha256':review_hashes[(benchmark,model)],
                'meaning':'Source family role only; no full-ROI position or causal classification.'}
            annotation_data=json.dumps(annotation_doc,indent=2,allow_nan=False).encode()+b'\n'
            annotation_path=target/'annotated.json.gz'
            annotation_path.write_bytes(gzip.compress(annotation_data,mtime=0))
            records=annotated
            annotated_selected = True
        # Preserve the best qualified capture when a later follow-up fails or
        # intentionally samples a narrower phase. Latest attempt status is kept
        # separately; raw attempts are never removed or mixed with selected data.
        score=(int(effective_status=='profile_process_completed' and not quality),
               int(bool(prov.get('retained_window_expected_count_confirmed'))),
               int(bool(prov.get('selection_expected_launch_count_matches'))),
               int(any(row.get('tensor_convention_enabled') and row.get('tensor_flops',0) for row in records)),
               int(bool(prov.get('additional_current_profile_input_bindings'))),
               int(bool(prov.get('filesystem_write_guard'))),
               int(bool(prov.get('profile_device_identity_verified'))),
               len(records),batch)
        key=(benchmark,model)
        if key in best_scores and score<best_scores[key]:
            dump(path,ledger)
            continue
        best_scores[key]=score
        regimes=collections.Counter()
        for record in records:
            for precision,point in record['scalar_precision_projections'].items():
                if point['executed_scalar_flops'] is not None and point['executed_scalar_flops']>0:
                    regimes[precision+':'+point['roofline_regime']]+=1
            if record.get('tensor_convention_enabled') and record.get('tensor_flops',0):
                regimes['tensor_fp16_inputs_fp32_accumulation:'+record['tensor_precision_projections']['dram']['roofline_regime']]+=1
        state.update(profiler_method=prov.get('profiler_method'),
            profiler_version=prov.get('profiler_version'),profiling_status=effective_status,
            measurement_method=(ALLREDUCE_METHOD if prov.get('allreduce_two_device_method') else 'ordinary_collector_profile_see_scope'),
            profile_method_admission=method_admission,
            process_unit_accounting=(method_admission.get('process_unit_accounting') if method_admission else None),
            profile_provenance_sha256=sha(provpath.read_bytes()),
            profile_applicability_review=({'path':str(applicability_path.relative_to(root)),
                'sha256':applicability_sha,'entry':benchmark+'/'+model,
                'reason':runtime_review['scope'],'parent_provenance_sha256':runtime_review['parent_profile']['sha256']}
                if runtime_applicable else None),
            selected_capture_policy='Qualified completion, retained-trial count proof, selector-count proof, typed tensor approval, source binding, write guard, PCI verification, kernel observations, lexical batch tie-break; latest attempt remains separate.',
            selected_capture_score=list(score),
            profile_process_status=prov['status'],
            profile_quality_review=({'path':str(quality_path.relative_to(root)),'sha256':quality_sha,'entry':quality} if quality else None),
            profile_ids=[str(target.relative_to(root))],
            profile_evidence_manifest=manifest,
            sampled_kernel_count=len(records),
            source_context_bound_before_after=bool(prov.get('additional_current_profile_input_bindings')),
            sampled_kernel_precision_regimes=dict(regimes),
            sampled_kernel_predicted_resource_limits=({key.replace('memory_side','memory_bound_roofline_prediction').replace('compute_side','compute_bound_roofline_prediction').replace('near_ridge','ridge_transition'):count for key,count in regimes.items()} if effective_status=='profile_process_completed' and not quality and benchmark not in ('zerocopy','halo-finder') else {}),
            dram_roofline_applicability=('not_applicable_host_mapped_primary_data_requires_PCIe_tier' if benchmark in ('zerocopy','halo-finder') else 'HBM_tier_projection_only_primary_memory_domain_requires_source_review'),
            roi_coverage=('allreduce_method_validation_hold_no_publishable_positions' if method_hold else
                          'single_source_marked_rank1_private_add_not_collective_ROI' if prov.get('allreduce_two_device_method') else
                          'not_applicable_no_profiled_device_kernels_source_reviewed' if runtime_applicable else 'not_collected_no_derived_counter_report' if not observation.is_file() else
                          'source_counted_retained_trial2_scientific_family_not_elapsed_ROI' if prov.get('retained_window_expected_count_confirmed') else
                          'partial_source_candidates_only_ROI_unresolved' if review and review.get('_candidate_review_only') else
                          'partial_source_family_reviewed_not_complete_ROI' if review else
                          'partial_first_specialization_geometry_unreviewed'),
            roofline_position=None,roofline_region=None,observed_bottleneck=None,
            reason=prov['selector_scope']+'; whole elapsed-ROI placement and causal bottleneck require separate review.')
        if observation.exists() and not method_hold:
            selected=target/('annotated.json.gz' if annotated_selected else 'derived.json.gz')
            state['kernel_observations']={'path':str(selected.relative_to(root)),
                'sha256':sha(selected.read_bytes()),
                'compression':'gzip','uncompressed_sha256':sha(gzip.decompress(selected.read_bytes()))}
        else:
            state.pop('kernel_observations',None)
        dump(path,ledger);imported.append((benchmark,model))
    slots=[];points=[];tensor_points=[];kernel_audit=[];hierarchical=[];statuses=collections.Counter();latest_statuses=collections.Counter();names=set();attempt_count=0
    for path in sorted((study/'benchmarks').glob('*.json')):
        ledger=json.loads(path.read_text())
        for model,state in ledger['models'].items():
            status=state['profiling_status'];statuses[status]+=1
            attempt_count+=len(state.get('profile_attempts',[]))
            if state.get('latest_profile_attempt'):latest_statuses[state['latest_profile_attempt']['status']]+=1
            if state.get('sampled_kernel_count',0)>0:names.add(ledger['benchmark'])
            slots.append({'benchmark':ledger['benchmark'],'model':model,
                'configuration_id':(ledger.get('accepted_named_contract')or{}).get('configuration_id'),
                'contract_outcome':(ledger.get('accepted_named_contract')or{}).get('outcome'),
                'reference_qualification':(ledger.get('accepted_named_contract')or{}).get('qualification'),
                'corrected_native':(ledger.get('accepted_named_contract')or{}).get('corrected_native'),
                'known_native_defect':(ledger.get('accepted_named_contract')or{}).get('known_defect'),
                'revised_acceptance':(ledger.get('accepted_named_contract')or{}).get('revised_acceptance'),
                'profiling_status':status,'plan_status':state.get('profile_plan_status'),
                'latest_attempt_status':state.get('latest_profile_attempt',{}).get('status'),
                'dram_roofline_applicability':state.get('dram_roofline_applicability','unreviewed'),
                'sampled_precision_regime_counts':json.dumps(state.get('sampled_kernel_precision_regimes',{}),sort_keys=True),
                'sampled_resource_prediction_counts':json.dumps(state.get('sampled_kernel_predicted_resource_limits',{}),sort_keys=True),
                'classification_scope':'Kernel samples only; whole elapsed ROI and causal bottleneck require separate evidence.',
                'measurement_method':state.get('measurement_method','ordinary_collector_profile_see_scope'),
                'method_admission_status':(state.get('profile_method_admission')or{}).get('status'),
                'process_unit_accounting':state.get('process_unit_accounting'),
                'selected_capture_policy':state.get('selected_capture_policy'),
                'profiler_method_id':(state.get('profiler_method')or{}).get('method_id','legacy2022_recorded_or_identity_incomplete'),
                'profiler_version':state.get('profiler_version'),
                'method_review_sha256':(state.get('profiler_method')or{}).get('review_sha256'),
                'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d',
                'primary_unprofiled_mean_seconds':(ledger.get('accepted_named_contract')or{}).get('models',{}).get(model,{}).get('mean'),
                'kernel_count':state.get('sampled_kernel_count',0),
                'roi_coverage':state['roi_coverage'],
                'whole_ROI_roofline_region':state.get('roofline_region'),
                'observed_bottleneck':state.get('observed_bottleneck'),
                'evidence_manifest':state.get('profile_evidence_manifest')})
            obs=state.get('kernel_observations')
            if not obs or obs.get('compression')!='gzip':continue
            derived=json.loads(gzip.decompress((root/obs['path']).read_bytes()))
            for k in derived['records']:
                common={'profiler_method_id':slots[-1]['profiler_method_id'],
                        'profiler_version':slots[-1]['profiler_version'],
                        'method_review_sha256':slots[-1]['method_review_sha256'],
                        'benchmark':ledger['benchmark'],'model':model,
                        'configuration_id':slots[-1]['configuration_id'],
                        'profiling_status':status,
                        'measurement_method':state.get('measurement_method','ordinary_collector_profile_see_scope'),
                        'dram_roofline_applicability':state.get('dram_roofline_applicability','unreviewed'),
                        'gpu_uuid':slots[-1]['gpu_uuid'],
                        'kernel_id':k['identity']['ID'],'kernel':k['identity']['Kernel Name'],
                        'process_id':k['identity'].get('Process ID'),'context_id':k['identity'].get('Context'),
                        'stream_id':k['identity'].get('Stream'),
                        'grid':k['identity'].get('Grid Size'),'block':k['identity'].get('Block Size'),
                        'stage':k['scope_context'].get('ROI_stage','unassigned'),
                        'semantic_family_in_primary_ROI':k['scope_context'].get('semantic_family_in_primary_ROI'),
                        'sample_phase':k['scope_context'].get('sample_phase','unreviewed'),
                        'sample_in_primary_timed_invocation':k['scope_context'].get('sample_in_primary_timed_invocation'),
                        'retained_trial':k['scope_context'].get('retained_trial'),
                        'phase_proof':k['scope_context'].get('phase_proof'),
                        'ROI_review_sha256':k['scope_context'].get('ROI_review_sha256'),
                        'roi_coverage':state['roi_coverage'],
                        'duration_seconds':k['duration_seconds'],
                        'dram_bytes':k['dram_bytes'],'L2_bytes':k['L2_transaction_bytes'],
                        'evidence_manifest':state.get('profile_evidence_manifest')}
                kernel_audit.append({**common,'precision_class':k['precision_class'],
                    **{p+'_'+op+'_thread_instructions':n for p,v in k['raw_scalar_predicated_on_thread_instruction_counts'].items()for op,n in v.items()},
                    'tensor_pipe_instructions':k['tensor_pipe_instructions'],
                    'tensor_flops':k['tensor_flops'],
                    'integer_thread_instructions':k.get('observed_integer_predicated_on_thread_instructions'),
                    'SM_utilization_pct':k.get('measured_utilization_pct',{}).get('SM'),
                    'DRAM_utilization_pct':k.get('measured_utilization_pct',{}).get('DRAM'),
                    'observed_bottleneck':'unknown','warnings':'; '.join(k['warnings'])})
                for tier, projection in k.get('tensor_precision_projections',{}).items():
                    if k.get('tensor_convention_enabled') and projection.get('executed_flops',0):
                        tensor_points.append({**common,'precision':'tensor_fp16_inputs_fp32_accumulation',
                            'precision_class':k['precision_class'],
                            'tensor_precision_mode':k['tensor_target_precision_scope'],
                            'selected_target_SASS_observed':k['scope_context'].get('selected_target_SASS_observed'),
                            'tensor_flops_per_instruction':k['tensor_flops_per_instruction'],
                            **projection})
                for tier,precision_points in k.get('scalar_precision_projections_by_memory_tier',{}).items():
                    for precision,p in precision_points.items():
                        n=p.get('executed_scalar_flops',p.get('executed_flops'))
                        if n is not None and n>0:
                            hierarchical.append({**common,'precision':precision,'memory_tier':tier,
                                'executed_flops':n,'arithmetic_intensity_flop_per_byte':p.get('arithmetic_intensity_flop_per_byte',p.get('dram_arithmetic_intensity_flop_per_byte')),
                                'achieved_flops_per_second':p['achieved_flops_per_second'],
                                'intensity_over_ridge':p['intensity_over_ridge'],'roofline_regime':p['roofline_regime'],
                                'projection_only':p['projection_only'],'causal_bottleneck':'unknown'})
                for precision,p in k['scalar_precision_projections'].items():
                    if p['executed_scalar_flops'] is None or p['executed_scalar_flops']==0:continue
                    points.append({'profiler_method_id':slots[-1]['profiler_method_id'],
                        'profiler_version':slots[-1]['profiler_version'],
                        'method_review_sha256':slots[-1]['method_review_sha256'],
                        'benchmark':ledger['benchmark'],'model':model,
                        'configuration_id':slots[-1]['configuration_id'],
                        'profiling_status':status,
                        'measurement_method':state.get('measurement_method','ordinary_collector_profile_see_scope'),
                        'dram_roofline_applicability':state.get('dram_roofline_applicability','unreviewed'),
                        'kernel_id':k['identity']['ID'],'kernel':k['identity']['Kernel Name'],
                        'process_id':k['identity'].get('Process ID'),'context_id':k['identity'].get('Context'),
                        'stream_id':k['identity'].get('Stream'),
                        'grid':k['identity'].get('Grid Size'),'block':k['identity'].get('Block Size'),
                        'precision':precision,'precision_class':k['precision_class'],
                        'stage':k['scope_context'].get('ROI_stage','unassigned'),
                        'semantic_family_in_primary_ROI':k['scope_context'].get('semantic_family_in_primary_ROI'),
                        'sample_phase':k['scope_context'].get('sample_phase','unreviewed'),
                        'sample_in_primary_timed_invocation':k['scope_context'].get('sample_in_primary_timed_invocation'),
                        'retained_trial':k['scope_context'].get('retained_trial'),
                        'phase_proof':k['scope_context'].get('phase_proof'),
                        'ROI_review_sha256':k['scope_context'].get('ROI_review_sha256'),
                        'roi_coverage':state['roi_coverage'],
                        'duration_seconds':k['duration_seconds'],
                        'dram_bytes':k['dram_bytes'],
                        'L2_bytes':k['L2_transaction_bytes'],
                        **p,'evidence_manifest':state.get('profile_evidence_manifest')})
    for filename,rows in [('model_audit.csv',slots),('kernel_audit.csv',kernel_audit),
                          ('kernel_roofline_points.csv',points),('kernel_hierarchical_points.csv',hierarchical),
                          ('kernel_tensor_roofline_points.csv',tensor_points)]:
        if not rows:continue
        temporary=study/(filename+'.new')
        with temporary.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
        temporary.replace(study/filename)
    progress={'schema_version':1,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'active','benchmark_population':537,'model_slots':2148,
        'status_counts':dict(statuses),'names_with_any_kernel_observation':len(names),
        'names_with_completed_qualified_kernel_observations':len({x['benchmark']for x in slots if x['profiling_status']=='profile_process_completed' and x['kernel_count']>0}),
        'names_with_all_four_completed_profiles':sum(all(x['profiling_status']=='profile_process_completed'for x in slots if x['benchmark']==name)for name in {x['benchmark']for x in slots}),
        'names_with_all_four_completed_kernel_observations':sum(all(x['profiling_status']=='profile_process_completed' and x['kernel_count']>0 for x in slots if x['benchmark']==name)for name in {x['benchmark']for x in slots}),
        'status_count_scope':'Best selected qualified capture/status per model slot; latest attempts and all historical failures remain separate.',
        'latest_attempt_status_counts':dict(latest_statuses),'retained_profile_attempts':attempt_count,
        'scalar_precision_kernel_projection_rows':len(points),
        'kernel_observations':len(kernel_audit),
        'hierarchical_scalar_projection_rows':len(hierarchical),
        'typed_tensor_memory_tier_projection_rows':len(tensor_points),
        'whole_ROI_classifications':sum(bool(x['whole_ROI_roofline_region'])for x in slots),
        'measured_causal_bottleneck_classifications':sum(bool(x['observed_bottleneck'])for x in slots),
        'interpretation':'Kernel samples and partial coverage only; inventory, profile success and whole-benchmark diagnosis are distinct.',
        'next':'Continue safe collection, blocked plan review, source/launch ROI mapping, hierarchical/precision validation and paper analysis.'}
    dump(study/'PROGRESS.json',progress)
    print(json.dumps(progress,indent=2))


if __name__=='__main__':
    with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as metadata_lock:
        fcntl.flock(metadata_lock,fcntl.LOCK_EX)
        main()
