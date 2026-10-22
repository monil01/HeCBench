from pathlib import Path
import hashlib,json,difflib
O=Path(__file__).resolve().parent;P=Path('/home/imo/HeCBench_julia/tools/v100_roofline_collect.py');original=P.read_text();assert hashlib.sha256(P.read_bytes()).hexdigest()=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116';changes=[];s=original
def replace(before,after,label):
 global s
 assert s.count(before)==1,(label,s.count(before));s=s.replace(before,after);changes.append({'label':label,'before':before,'after':after})
replace('import math\n','import math\nimport select\n','stdlib_pidfd_poll_only')
marker='def run(command, env, cwd, log, timeout, reserve, guard=None,monitor_gpu=False,artifact_policy=None):\n'
replace(marker,(O/'ownership_helper_source.py').read_text()+'\n\n'+marker,'inline_generation_cache')
replace('        stop_reason = None\n        external_activity=[];last_gpu_check=0\n','''        ownership=None;ownership_bootstrap_error=None
        if monitor_gpu:
            try:ownership=ProcessGenerationOwnership(process.pid)
            except (OSError,ValueError,IndexError,RuntimeError) as error:ownership_bootstrap_error=repr(error)
        stop_reason = 'ownership_generation_bootstrap_failed' if ownership_bootstrap_error else None
        unresolved_ownership=[];leader_signal_identity_hold=False
        external_activity=[];last_gpu_check=0
''','bootstrap_owned_Popen_generation')
replace('        while process.poll() is None:\n            if time.monotonic() - began > timeout:\n','''        while process.poll() is None:
            if ownership is not None and not stop_reason:
                try:ownership.refresh()
                except (OSError,ValueError,IndexError,RuntimeError) as error:
                    stop_reason='ownership_cache_refresh_failed'
                    unresolved_ownership.append({'error':repr(error),'observed_at':utc()})
            if time.monotonic() - began > timeout:
''','capture_live_session_generations_between_GPU_queries')
replace("                    external_activity.append({'monitor_error':str(error),'observed_at':utc()})\n","                    unresolved_ownership.append({'monitor_error':str(error),'observed_at':utc()})\n",'GPUqueryerror_is_not_foreign_process_proof')
a=s.index('                for item in observed:\n');b=s.index('            if stop_reason:\n',a);before=s[a:b]
replace(before,'''                for item in observed:
                    result=(ownership.classify(item['pid']) if ownership is not None else
                            {'class':'ownership_bootstrap_unresolved','pid':item['pid']})
                    event={**item,'observed_at':utc(),'ownership':result}
                    if ownership is not None:ownership.events.append(event)
                    if result['class']=='live_generation_owned':
                        continue
                    if result['class']=='live_process_outside_owner_uid':external_activity.append(event)
                    else:unresolved_ownership.append(event)
                if external_activity:stop_reason='external_gpu_activity_detected_own_profile_stopped'
                elif unresolved_ownership and not stop_reason:stop_reason='GPU_process_generation_unresolved_own_profile_stopped'
''','fail_closed_distinct_foreign_vs_unresolved_prior_own_teardown')
replace("                if os.getsid(process.pid) != process.pid:\n                    raise RuntimeError('Own child session guard failed')\n                os.killpg(process.pid, signal.SIGTERM)","""                if monitor_gpu:
                    try:safe_leader=ownership is not None and ownership.cached_live(process.pid)
                    except (OSError,ValueError,IndexError,RuntimeError):safe_leader=False
                    if not safe_leader:
                        leader_signal_identity_hold=True
                        stop_reason='own_leader_generation_unresolved_no_signal'
                        break
                if os.getsid(process.pid) != process.pid:
                    raise RuntimeError('Own child session guard failed')
                os.killpg(process.pid, signal.SIGTERM)""",'generation_proof_before_own_group_TERM')
replace("                except subprocess.TimeoutExpired:\n                    if os.getsid(process.pid) != process.pid:","""                except subprocess.TimeoutExpired:
                    if monitor_gpu:
                        try:safe_leader=ownership is not None and ownership.cached_live(process.pid)
                        except (OSError,ValueError,IndexError,RuntimeError):safe_leader=False
                        if not safe_leader:
                            leader_signal_identity_hold=True
                            stop_reason='own_leader_generation_unresolved_no_signal'
                            break
                    if os.getsid(process.pid) != process.pid:""",'generation_proof_before_own_group_KILL')
replace('                os.killpg(process.pid, signal.SIGTERM)\n', '''                if monitor_gpu:
                    result=ownership.signal_fresh_session(signal.SIGTERM)
                    if result['rejected']:
                        leader_signal_identity_hold=True
                        stop_reason='own_signal_generation_unresolved_no_unknown_signals'
                        break
                else:os.killpg(process.pid, signal.SIGTERM)
''','atomic_pidfd_TERM_for_proved_own_session_generations')
replace('                    os.killpg(process.pid, signal.SIGKILL)\n', '''                    if monitor_gpu:
                        result=ownership.signal_fresh_session(signal.SIGKILL)
                        if result['rejected']:
                            leader_signal_identity_hold=True
                            stop_reason='own_signal_generation_unresolved_no_unknown_signals'
                            break
                    else:os.killpg(process.pid, signal.SIGKILL)
''','atomic_pidfd_KILL_for_proved_own_session_generations')
replace("                if os.getsid(process.pid) != process.pid:\n                    raise RuntimeError('Own child session guard failed')\n", "                if not monitor_gpu and os.getsid(process.pid) != process.pid:\n                    raise RuntimeError('Own child session guard failed')\n",'monitored_signals_use_pidfd_session_proof_without_numeric_getsid_race')
replace("                    if os.getsid(process.pid) != process.pid:\n                        raise RuntimeError('Own child session guard failed')\n", "                    if not monitor_gpu and os.getsid(process.pid) != process.pid:\n                        raise RuntimeError('Own child session guard failed')\n",'monitored_KILL_uses_pidfd_session_proof_without_numeric_getsid_race')
replace('        code = process.wait()\n','        code = process.poll() if leader_signal_identity_hold else process.wait()\n','do_not_block_or_signal_unknown_generation')
a=s.index('        quiescence_started=time.monotonic();descendant_cleanup=[]\n');b=s.index("    return {'returncode': code",a);before=s[a:b]
replace(before,'''        quiescence_started=time.monotonic();descendant_cleanup=[]
        quiescence_unknowns=[];own_session_quiescent=not leader_signal_identity_hold
        while own_session_quiescent:
            try:members=own_session_members(process.pid)
            except (OSError,ValueError,IndexError,RuntimeError) as error:
                if not monitor_gpu:raise
                own_session_quiescent=False;quiescence_unknowns.append({'error':repr(error)});break
            if ownership is not None:
                try:members=sorted(set(members)|{pid for pid,r in ownership.records.items() if ownership.fd_live(r)})
                except (OSError,ValueError,IndexError,RuntimeError) as error:
                    own_session_quiescent=False;quiescence_unknowns.append({'error':repr(error)});break
            if not members:break
            if monitor_gpu and time.monotonic()-quiescence_started>20:
                own_session_quiescent=False
                quiescence_unknowns.append({'members':members,'reason':'bounded_quiescence_not_established'})
                break
            if time.monotonic()-quiescence_started>10:
                if stop_reason is None:stop_reason='own_descendants_outlived_profile_leader'
                for pid in members:
                    try:
                        path=Path('/proc')/str(pid)
                        if path.stat().st_uid!=os.getuid() or os.getsid(pid)!=process.pid:
                            raise RuntimeError('Own session cleanup identity changed')
                        if monitor_gpu:
                            result=ownership.classify(pid) if ownership is not None else {'class':'ownership_bootstrap_unresolved','pid':pid}
                            if result['class']!='live_generation_owned' or not ownership.cached_live(pid):
                                quiescence_unknowns.append(result);own_session_quiescent=False
                                break
                        if monitor_gpu:ownership.signal_cached_member(pid,signal.SIGKILL)
                        else:os.kill(pid,signal.SIGKILL)
                        descendant_cleanup.append(pid)
                    except ProcessLookupError:pass
                    except (OSError,ValueError,IndexError,RuntimeError) as error:
                        if not monitor_gpu:raise
                        quiescence_unknowns.append({'pid':pid,'error':repr(error)})
                        own_session_quiescent=False;break
            if own_session_quiescent:time.sleep(.5)
        try:ownership_evidence=ownership.snapshot() if ownership is not None else {'bootstrap_error':ownership_bootstrap_error}
        except (OSError,ValueError,IndexError,RuntimeError) as error:
            ownership_evidence={'snapshot_error':repr(error)};stop_reason='ownership_evidence_snapshot_failed'
        finally:
            if ownership is not None:ownership.close()
        if not own_session_quiescent:stop_reason='own_session_generation_quiescence_unresolved_hold'
''','retain_generation_cache_through_safe_quiescence_no_unknown_signals')
needle_hold="        try:ownership_evidence=ownership.snapshot() if ownership is not None else {'bootstrap_error':ownership_bootstrap_error}\n"
replace(needle_hold, '''        ownership_quiescence_hold=None
        if monitor_gpu and not own_session_quiescent:
            # Both monitored callers are inside collect's existing GPU flock.
            # Never unwind that lock while a known/potential own writer remains.
            # No signal, GPU query, data hash/freeze or deletion occurs here.
            ownership_quiescence_hold={'started_at':utc(),'GPU_lock_release_permitted':False,
                'scope':'Unbounded safety/quiescence hold, outside process deadline; ambiguous interval is never admitted'}
            stop_reason='ownership_quiescence_was_unresolved_no_admission'
            while True:
                pending=[];errors=[]
                try:
                    if process.poll() is None:pending.append({'pid':process.pid,'basis':'fresh_Popen_child_not_reaped'})
                except (OSError,ValueError,IndexError,RuntimeError) as error:errors.append(repr(error))
                try:
                    for pid in own_session_members(process.pid):pending.append({'pid':pid,'basis':'fresh_session_membership_not_yet_quiescent'})
                    if ownership is not None:
                        for pid,record in ownership.records.items():
                            if ownership.fd_live(record):pending.append({'pid':pid,'basis':'pinned_own_generation_alive'})
                except (OSError,ValueError,IndexError,RuntimeError) as error:errors.append(repr(error))
                # Unknown generation is never signaled or assumed foreign. A
                # newly reused PID may conservatively prolong this safety hold.
                for event in unresolved_ownership+quiescence_unknowns:
                    pid=event.get('pid')
                    if pid is None:continue
                    try:
                        (Path('/proc')/str(pid)).stat()
                        pending.append({'pid':pid,'basis':'recorded_unresolved_PID_still_exists_no_generation_admission'})
                    except (FileNotFoundError,ProcessLookupError):pass
                    except OSError as error:errors.append(repr(error))
                ownership_quiescence_hold.update(last_checked_at=utc(),pending=pending,errors=errors)
                if not pending and not errors:
                    own_session_quiescent=True
                    ownership_quiescence_hold.update(finished_at=utc(),GPU_lock_release_permitted=True,
                        outcome='quiescence_observed_after_hold_attempt_remains_failed')
                    code=process.wait()
                    try:dump(cwd.parent/'ownership_quiescence_hold.json',ownership_quiescence_hold)
                    except OSError:pass
                    break
                try:dump(cwd.parent/'ownership_quiescence_hold.json',ownership_quiescence_hold)
                except OSError:pass
                time.sleep(1)
        try:ownership_evidence=ownership.snapshot() if ownership is not None else {'bootstrap_error':ownership_bootstrap_error}
''','retain_GPU_flock_until_owned_and_recorded_unknown_live_writers_gone')
replace("            'external_gpu_activity':external_activity,\n","""            'external_gpu_activity':external_activity,
            'unresolved_GPU_ownership':unresolved_ownership,
            'process_generation_ownership':ownership_evidence,
            'own_session_quiescent':own_session_quiescent,
            'own_session_quiescence_unknowns':quiescence_unknowns,
            'ownership_quiescence_hold':ownership_quiescence_hold,
""",'persist_ownership_science_separate_from_process_outcome')
marker="                                        timeout, reserve_mib*1024**2,guard,monitor_gpu=True,artifact_policy=artifact_policy)\n"
replace(marker,marker+"            if not provenance['profile'].get('own_session_quiescent',False):\n                raise RuntimeError('Own-session generation/quiescence unresolved: no counter freeze or scratch cleanup')\n",'mandatory_quiescence_gate_before_counter_snapshots')
a=s.index('        if artifact_policy:\n',s.index('    finally:\n',s.index("    except Exception as error:\n        if provenance['status']")));b=s.index("        provenance['finished_at'] = utc()\n",a);body=s[a:b]
replace(body,"""        ownership_hold=any(isinstance(provenance.get(k),dict) and provenance[k].get('own_session_quiescent') is False
                           for k in ('profile','pre_profile_checks','build'))
        if ownership_hold:
            provenance['status']='collection_failed_ownership_quiescence_unresolved'
            provenance['scratch_retained_unfrozen_ownership_hold']=True
            provenance['scratch_freeze_cleanup']='Not performed: live/unknown session generation prevents stable evidence or disposable-scratch cleanup'
        else:
"""+''.join('    '+line if line.strip()else line for line in body.splitlines(keepends=True)),'no_manifest_freeze_or_cleanup_of_unquiescent_ownership_hold')
replace("            'wall_seconds': time.monotonic() - began, 'log_sha256': sha(log),\n", "            'wall_seconds': time.monotonic() - began, 'log_sha256': sha(log) if own_session_quiescent else None,\n            'execution_log_freeze_status':'quiescent_hash' if own_session_quiescent else 'unfrozen_live_writer_ownership_hold',\n",'never_hash_log_while_unquiescent_writer_may_live')
rev=s
for c in reversed(changes):assert rev.count(c['after'])==1,c['label'];rev=rev.replace(c['after'],c['before'])
assert rev==original
candidate=O/'v100_roofline_collect_ownership_proposal.py';candidate.write_text(s);(O/'minimal_source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(P),tofile=str(candidate))))
(O/'source_inverse_proof.json').write_text(json.dumps({'original':{'path':str(P),'sha256':hashlib.sha256(P.read_bytes()).hexdigest()},'candidate':{'path':str(candidate),'sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()},'exact_full_byte_inverse_equal':rev.encode()==P.read_bytes(),'changes':changes,'candidate_import_compile_execute_tests':False,'historical_outcomes_not_modified':True},indent=2)+'\n');print('candidateSHA',hashlib.sha256(candidate.read_bytes()).hexdigest(),'changes',len(changes))
