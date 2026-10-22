from pathlib import Path
import json,hashlib,difflib
out=Path(__file__).parent
old=out.parent/'own_teardown_guard_proposal_v4/v100_roofline_collect_ownership_proposal.py'
base=old.read_text(); text=base; changes=[]
def replace(label,before,after):
 global text
 assert text.count(before)==1,(label,text.count(before))
 text=text.replace(before,after);changes.append(dict(label=label,before=before,after=after))
scanner='''def monitored_session_scan(session, tracked=()):
    """Membership evidence only; errors never authorize ownership or signals.

    Known other sessions/UIDs are excluded before full identity reads. An unknown
    candidate that cannot be excluded is retained until confirmed /proc absence.
    """
    result={'members':[], 'unresolved':[], 'disappeared':[], 'excluded':0}
    tracked=set(tracked)
    try:
        pids={int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()}|tracked
    except OSError as error:
        result['unresolved'].append({'pid':None,'stage':'proc_enumeration','error':repr(error)})
        return result
    for pid in sorted(pids):
        path=Path('/proc')/str(pid)
        sid=None;uid=None;errors=[]
        try:sid=os.getsid(pid)
        except OSError as error:errors.append({'stage':'getsid','error':repr(error)})
        try:uid=path.stat().st_uid
        except OSError as error:errors.append({'stage':'directory_stat','error':repr(error)})
        if pid not in tracked and ((sid is not None and sid!=session) or (uid is not None and uid!=os.getuid())):
            result['excluded']+=1
            continue
        try:
            # Confirm directory absence explicitly; a missing stat/ns file while
            # the directory exists is unresolved, not a disappeared process.
            path.stat()
        except (FileNotFoundError,ProcessLookupError):
            result['disappeared'].append({'pid':pid,'basis':'proc_directory_absent','errors':errors})
            continue
        except OSError as error:errors.append({'stage':'absence_confirmation','error':repr(error)})
        if pid in tracked and ((sid is not None and sid!=session) or (uid is not None and uid!=os.getuid())):
            errors.append({'stage':'tracked_candidate_boundary_changed','error':'No generation transfer or detached-writer exclusion from numeric PID'})
        if not errors:
            try:
                raw=(path/'stat').read_text()
                fields=raw.rsplit(')',1)[1].split()
                if int(raw.split(' ',1)[0])!=pid or int(fields[3])!=sid:
                    raise RuntimeError('Session/PID changed during membership snapshot')
                if sid==session and uid==os.getuid() and fields[0] not in ('Z','X'):
                    result['members'].append(pid)
            except (OSError,ValueError,IndexError,RuntimeError) as error:
                errors.append({'stage':'membership_stat','error':repr(error)})
        if errors:
            try:path.stat()
            except (FileNotFoundError,ProcessLookupError):
                result['disappeared'].append({'pid':pid,'basis':'proc_directory_absent_after_error','errors':errors})
                continue
            except OSError as error:errors.append({'stage':'final_absence_confirmation','error':repr(error)})
            result['unresolved'].append({'pid':pid,'session':sid,'uid':uid,'errors':errors})
    return result


'''
replace('monitored_only_scanner','class ProcessGenerationOwnership:\n',scanner+'class ProcessGenerationOwnership:\n')
replace('tracked_unknown_candidates','        self.records = {}\n','        self.records = {}\n        self.scan_unknown_pids = set()\n')
start=text.index('    def refresh(self):\n');end=text.index('    def signal_fresh_session',start)
replace('strict_refresh',text[start:end],'''    def scan_members(self):
        scan=monitored_session_scan(self.leader, set(self.records)|self.scan_unknown_pids)
        self.scan_unknown_pids.update(e['pid'] for e in scan['unresolved'] if e.get('pid') is not None)
        self.scan_unknown_pids.difference_update(e['pid'] for e in scan['disappeared'])
        if scan['unresolved'] or scan['disappeared']:
            self.events.append({'kind':'strict_monitored_session_scan','scan':scan,'observed_at':utc()})
        return scan

    def refresh(self):
        scan=self.scan_members()
        if scan['unresolved']:
            raise RuntimeError('Relevant monitored session scan unresolved: '+repr(scan['unresolved']))
        for pid in sorted(set(scan['members'])|set(self.records)):
            try:
                if pid in self.records:
                    if self.fd_live(self.records[pid]):self.cached_live(pid)
                    continue
                identity=self.read_identity(pid)
                proof=self.live_proof(identity)
                if proof is None:raise RuntimeError('Live session member ownership proof unresolved')
                self.pin(identity,proof)
            except (OSError,ValueError,IndexError,RuntimeError) as error:
                # Retain the numeric candidate for quiescence only, never as an
                # ownership token or permission to signal a later generation.
                self.scan_unknown_pids.add(pid)
                self.events.append({'kind':'refresh_identity_unresolved','pid':pid,'error':repr(error),'observed_at':utc()})
                raise RuntimeError('Relevant refresh identity unresolved: '+str(pid)) from error

''')
replace('signal_strict_scan',"            members=own_session_members(self.leader)\n", "            scan=self.scan_members()\n            if scan['unresolved']:\n                return {'delivered':[],'rejected':scan['unresolved']}\n            members=scan['members']\n")
replace('snapshot_unknown_candidates',"                'GPU_observations':self.events,\n", "                'GPU_observations':self.events,\n                'unresolved_session_candidate_pids':sorted(self.scan_unknown_pids),\n")
replace('quiescence_monitored_branch','            try:members=own_session_members(process.pid)\n', '''            try:
                if monitor_gpu:
                    scan=ownership.scan_members() if ownership is not None else monitored_session_scan(process.pid)
                    if scan['unresolved']:
                        own_session_quiescent=False;quiescence_unknowns.extend(scan['unresolved']);break
                    members=scan['members']
                else:members=own_session_members(process.pid)
''')
replace('hold_strict_scan',"                    for pid in own_session_members(process.pid):pending.append({'pid':pid,'basis':'fresh_session_membership_not_yet_quiescent'})\n",'''                    tracked={event['pid'] for event in unresolved_ownership+quiescence_unknowns if event.get('pid') is not None}
                    if ownership is not None:tracked.update(ownership.records);tracked.update(ownership.scan_unknown_pids)
                    scan=monitored_session_scan(process.pid,tracked)
                    pending.extend({'pid':pid,'basis':'fresh_session_membership_not_yet_quiescent'} for pid in scan['members'])
                    pending.extend(dict(event,basis='strict_session_scan_unresolved_no_signal') for event in scan['unresolved'])
''')
candidate=out/'v100_roofline_collect_ownership_proposal.py';candidate.write_text(text)
restored=text
for change in reversed(changes):
 assert restored.count(change['after'])==1,change['label']
 restored=restored.replace(change['after'],change['before'])
assert restored.encode()==old.read_bytes()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
canonical=Path('/home/imo/HeCBench_julia/tools/v100_roofline_collect.py')
oldproof=json.loads((old.parent/'source_inverse_proof.json').read_text())
canonical_restore=restored
for change in reversed(oldproof['changes']):
 assert canonical_restore.count(change['after'])==1
 canonical_restore=canonical_restore.replace(change['after'],change['before'])
assert canonical_restore.encode()==canonical.read_bytes()
(out/'source_inverse_proof.json').write_text(json.dumps(dict(original=dict(path=str(old),sha256=sha(old)),candidate=dict(path=str(candidate),sha256=sha(candidate)),exact_v4_full_byte_inverse_equal=True,exact_C21_composed_inverse_equal=True,canonical_sha256=sha(canonical),changes=changes),indent=2)+'\n')
(out/'minimal_v4_to_v5.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(candidate))))
print(json.dumps(dict(candidate_sha256=sha(candidate),changes=len(changes),byte_inverse=True)))
