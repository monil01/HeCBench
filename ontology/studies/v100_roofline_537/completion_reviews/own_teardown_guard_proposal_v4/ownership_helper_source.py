class ProcessGenerationOwnership:
    """Live generation proof, never permission to admit a vanished numeric NVML PID.

    pidfds stay open until run's quiescence decision. A dead pidfd identifies the
    cached process that exited; it cannot identify a subsequent NVML numeric row.
    """
    def __init__(self, leader):
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('pidfd support required; no name/PID-only fallback')
        self.leader = leader
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.pid_namespace = Path('/proc/self/ns/pid').stat().st_ino
        self.records = {}
        self.events = []
        self.closed = False
        identity = self.read_identity(leader)
        if identity['uid'] != os.getuid() or identity['session'] != leader:
            raise RuntimeError('Fresh Popen root UID/session differs')
        self.root = self.pin(identity, {'kind':'fresh_Popen_start_new_session_anchor'})

    def read_identity(self, pid):
        path = Path('/proc') / str(pid)
        first = (path/'stat').read_text()
        left, fields = first.rsplit(')', 1)
        values = fields.split()
        record = {'pid':int(first.split(' ',1)[0]), 'ppid':int(values[1]),
                  'session':int(values[3]), 'start_ticks':values[19],
                  'state':values[0], 'uid':path.stat().st_uid,
                  'pid_namespace':(path/'ns/pid').stat().st_ino,
                  'raw_stat':first, 'observed_at':utc()}
        second = (path/'stat').read_text().rsplit(')',1)[1].split()
        if (record['pid'] != pid or record['start_ticks'] != second[19]
                or record['ppid'] != int(second[1]) or record['session'] != int(second[3])
                or record['uid'] != path.stat().st_uid
                or record['pid_namespace'] != self.pid_namespace):
            raise RuntimeError('Process identity changed during proc snapshot')
        return record

    @staticmethod
    def generation(identity):
        return (identity['pid'], identity['start_ticks'], identity['uid'], identity['pid_namespace'])

    @staticmethod
    def fd_live(record):
        watcher = select.poll()
        watcher.register(record['pidfd'], select.POLLIN | select.POLLHUP | select.POLLERR)
        return not watcher.poll(0)

    def pin(self, identity, proof):
        pid = identity['pid']
        previous = self.records.get(pid)
        if previous:
            if self.generation(previous['identity']) != self.generation(identity):
                raise RuntimeError('Numeric PID reuse; cached ownership never transfers')
            if not self.fd_live(previous):
                raise RuntimeError('Exited cached generation cannot be repinned')
            previous['last_live_identity'] = identity
            return previous
        fd = os.pidfd_open(pid, 0)
        try:
            os.set_inheritable(fd, False)
            second = self.read_identity(pid)
            record = {'identity':identity, 'last_live_identity':second,
                      'pidfd':fd, 'proof':proof, 'pinned_at':utc()}
            if (self.generation(identity) != self.generation(second)
                    or second['state'] in ('Z','X') or not self.fd_live(record)):
                raise RuntimeError('Generation exited or changed while pinning')
            self.records[pid] = record
            return record
        except BaseException:
            os.close(fd)
            raise

    def cached_live(self, pid):
        record = self.records.get(pid)
        if record is None or not self.fd_live(record):
            return False
        current = self.read_identity(pid)
        if (self.generation(current) != self.generation(record['identity'])
                or current['state'] in ('Z','X')):
            return False
        record['last_live_identity'] = current
        return self.fd_live(record)

    def live_proof(self, identity):
        if identity['uid'] != os.getuid():
            return None
        # The live pinned fresh root's session is an owned process boundary.
        if self.cached_live(self.leader) and identity['session'] == self.leader:
            return {'kind':'same_live_generation_session', 'anchor':self.root['identity'],
                    'member':identity}
        chain = [identity]
        seen = {identity['pid']}
        current = identity
        for unused in range(128):
            parent = current['ppid']
            if parent <= 1 or parent in seen:
                return None
            seen.add(parent)
            parent_identity = self.read_identity(parent)
            if (parent_identity['uid'] != os.getuid()
                    or int(parent_identity['start_ticks']) > int(current['start_ticks'])):
                return None
            chain.append(parent_identity)
            if self.cached_live(parent):
                return {'kind':'live_parent_chain_to_pinned_own_generation', 'chain':chain,
                        'anchor':self.records[parent]['identity']}
            current = parent_identity
        return None

    def classify(self, pid):
        prior = self.records.get(pid)
        try:
            identity = self.read_identity(pid)
            if prior:
                if self.generation(identity) != self.generation(prior['identity']):
                    return {'class':'PID_reuse_or_identity_change_hold', 'pid':pid,
                            'prior':prior['identity'], 'current':identity}
                if self.cached_live(pid):
                    return {'class':'live_generation_owned', 'pid':pid, 'identity':identity,
                            'proof':prior['proof']}
                return {'class':'prior_own_generation_exited_NVML_generation_unresolved',
                        'pid':pid, 'prior':prior['identity'], 'current':identity}
            proof = self.live_proof(identity)
            if proof is None:
                return {'class':('live_process_outside_owner_uid' if identity['uid'] != os.getuid()
                                 else 'live_unproven_ancestry_generation_unresolved'),
                        'pid':pid, 'identity':identity}
            record = self.pin(identity, proof)
            if not self.cached_live(pid):
                raise RuntimeError('Owned generation exited before classification')
            return {'class':'live_generation_owned', 'pid':pid, 'identity':identity,
                    'proof':record['proof']}
        except (OSError, ValueError, IndexError, RuntimeError) as error:
            return {'class':('prior_own_generation_NVML_generation_unresolved' if prior
                             else 'unknown_NVML_generation_unresolved'),
                    'pid':pid, 'error':repr(error),
                    'prior':prior['identity'] if prior else None}

    def refresh(self):
        # Record real live descendants before teardown, even between GPU queries.
        # Failed proc reads do not register ownership or transfer an old PID token.
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                identity = self.read_identity(int(path.name))
                if identity['uid'] != os.getuid():
                    continue
                if int(path.name) in self.records:
                    self.cached_live(int(path.name))
                elif identity['session'] == self.leader:
                    proof = self.live_proof(identity)
                    if proof is not None:
                        self.pin(identity, proof)
            except (OSError, ValueError, IndexError, RuntimeError):
                continue

    def signal_fresh_session(self, sig):
        # pidfd_send_signal targets the pinned task generation, never a numeric
        # PID/PGID that might have been reused between validation and delivery.
        targets=[]
        rejected=[]
        try:
            self.refresh()
            members=own_session_members(self.leader)
        except (OSError,ValueError,IndexError,RuntimeError) as error:
            return {'delivered':[],'rejected':[{'error':repr(error),'class':'signal_scan_unresolved'}]}
        for pid in members:
            try:
                result=self.classify(pid)
                if result['class']!='live_generation_owned':
                    rejected.append(result)
                    continue
                current=self.read_identity(pid)
                if current['session']!=self.leader or not self.cached_live(pid):
                    rejected.append({'pid':pid,'class':'signal_session_or_generation_unresolved'})
                    continue
                targets.append(pid)
            except (OSError,ValueError,IndexError,RuntimeError) as error:
                rejected.append({'pid':pid,'error':repr(error)})
        delivered=[]
        for pid in sorted(targets, key=lambda p:p==self.leader):
            try:
                current=self.read_identity(pid)
                if current['session']!=self.leader or not self.cached_live(pid):
                    rejected.append({'pid':pid,'class':'signal_identity_changed'})
                    continue
                signal.pidfd_send_signal(self.records[pid]['pidfd'],sig,None,0)
                delivered.append(pid)
            except ProcessLookupError:
                pass
            except (OSError,ValueError,IndexError,RuntimeError) as error:
                rejected.append({'pid':pid,'error':repr(error)})
        self.events.append({'kind':'owned_fresh_session_pidfd_signal',
                            'signal':int(sig),'delivered_generations':delivered,
                            'rejected_no_signal':rejected,'observed_at':utc()})
        return {'delivered':delivered,'rejected':rejected}

    def signal_cached_member(self, pid, sig):
        current=self.read_identity(pid)
        if current['session']!=self.leader or not self.cached_live(pid):
            raise RuntimeError('Cleanup session/generation unproved; no signal')
        signal.pidfd_send_signal(self.records[pid]['pidfd'],sig,None,0)

    def snapshot(self):
        return {'policy':'live_generation_only_no_dead_numeric_PID_admission',
                'boot_id':self.boot_id, 'pid_namespace':self.pid_namespace,
                'leader':self.leader,
                'generations':[{'identity':r['identity'],
                                'last_live_identity':r['last_live_identity'],
                                'proof':r['proof'], 'pinned_at':r['pinned_at'],
                                'pidfd_live_at_snapshot':self.fd_live(r)}
                               for r in self.records.values()],
                'GPU_observations':self.events,
                'vanished_cached_PID_is_not_proven_foreign_or_admissibly_owned':True}

    def close(self):
        if not self.closed:
            for record in self.records.values():
                os.close(record['pidfd'])
            self.closed = True
