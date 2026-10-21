"""Hash-bound current537 display decoration. Never supplies graph scoring inputs.

Absent/unapproved authority disables decoration. Each failed entry is withheld
independently; original source records remain the sole returned fallback.
"""
import copy
import fcntl
import gzip
import hashlib
import json
import os
import stat
from pathlib import Path

SUBTREE = 'ontology/studies/v100_roofline_537/completion_reviews/current537_display_traits'
SOURCE_SUFFIXES = {'.jl', '.cu', '.cpp', '.c', '.h', '.hpp', '.py'}
ALLOWED_OVERRIDES = {'work_item_mapping', 'launch_geometry'}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def proof_sources(value):
    found = {}
    if isinstance(value, dict):
        digest = value.get('source_sha256') or value.get('sha256')
        path = value.get('path')
        if path and digest and Path(path).suffix in SOURCE_SUFFIXES:
            found[str(Path(path).resolve())] = digest
        for item in value.values():
            found.update(proof_sources(item))
    elif isinstance(value, list):
        for item in value:
            found.update(proof_sources(item))
    return found


class DisplayTraits:
    def __init__(self, root, population, review_sha256=None, metadata_lock_fd=None):
        self.root = Path(root).resolve()
        self.population = population
        self.borrowed_lock_fd = metadata_lock_fd
        self.overlay = None
        self.issues = []
        self.cache = {}
        self.inputs = {}
        self.authority = None
        if population != 537 or not review_sha256:
            self.issues.append({'scope':'authority', 'reason':'disabled_population_or_no_explicit_review_SHA'})
            return
        try:
            with self.lock():
                review_path = self.root/SUBTREE/'root_review.json'
                review = self.read(review_path, review_sha256)
                if review.get('status') != 'root_approved_current537_display_traits':
                    raise ValueError('Root display authority not approved')
                if review.get('population') != 537:
                    raise ValueError('Authority population mismatch')
                loader = review['loader']
                if loader['path'] != 'ontology/tools/current537_display_traits.py':
                    raise ValueError('Loader path mismatch')
                if sha(Path(__file__).read_bytes()) != loader['sha256']:
                    raise ValueError('Loader byte hash mismatch')
                self.inputs[str(Path(__file__).resolve())] = loader['sha256']
                overlay_ref = review['overlay']
                if overlay_ref['path'] != SUBTREE+'/overlay.json':
                    raise ValueError('Noncanonical overlay path')
                overlay = self.read(self.root/overlay_ref['path'], overlay_ref['sha256'])
                if overlay.get('status') not in {'PRIVATE_PROPOSED_DISPLAY_OVERLAY_NOT_ADOPTED','root_reviewed_display_overlay'}:
                    raise ValueError('Unknown overlay schema status')
                if overlay.get('schema_version') != 1:
                    raise ValueError('Unknown display overlay schema')
                if overlay['root_adoption'] != review['source_adoption']:
                    raise ValueError('Overlay/source adoption authority disagreement')
                adoption = self.read(self.root/review['source_adoption']['path'], review['source_adoption']['sha256'])
                if adoption.get('status') != 'source_annotations_adopted_zero_issue_integrity_and_postannotation_audits':
                    raise ValueError('Source annotation adoption not audited')
                self.overlay = overlay
                self.authority = {'path':str(review_path.relative_to(self.root)), 'sha256':review_sha256,
                                  'overlay':overlay_ref, 'source_adoption':review['source_adoption']}
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.issues.append({'scope':'authority', 'reason':str(error)})
            self.overlay = None

    def lock(self):
        # Existing read-only coordinator lock; no writes or lock-file creation.
        class Shared:
            def __enter__(inner):
                if self.borrowed_lock_fd is not None:
                    # Borrow the exact inherited open-file description. EX -> EX
                    # never downgrades the parent's lease; NB never waits on it.
                    actual = os.fstat(self.borrowed_lock_fd)
                    expected = os.stat('/tmp/hecbench-roofline-metadata.lock')
                    if (not stat.S_ISREG(actual.st_mode)
                            or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino)):
                        raise ValueError('Borrowed FD is not canonical metadata lock')
                    fcntl.flock(self.borrowed_lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    inner.file = None
                else:
                    inner.file = open('/tmp/hecbench-roofline-metadata.lock','r')
                    fcntl.flock(inner.file,fcntl.LOCK_SH)
            def __exit__(inner,*unused):
                # Never unlock/close the borrowed lease: parent still owns it.
                if inner.file is not None:
                    fcntl.flock(inner.file,fcntl.LOCK_UN)
                    inner.file.close()
        return Shared()

    def read(self, path, expected):
        path = Path(path)
        raw = path.read_bytes()
        if sha(raw) != expected:
            raise ValueError('SHA mismatch: '+str(path))
        self.inputs[str(path.resolve())] = expected
        if path.suffix == '.gz':
            raw = gzip.decompress(raw)
        return json.loads(raw)

    def unique(self, values, predicate):
        matches = [value for value in values if predicate(value)]
        if len(matches) != 1:
            raise ValueError('Missing or nonunique display entry')
        return matches[0]

    def capture(self, benchmark, model):
        self.read(self.root/self.authority['path'], self.authority['sha256'])
        overlay = self.read(self.root/self.authority['overlay']['path'], self.authority['overlay']['sha256'])
        capture = self.unique(overlay['selected_capture_bindings'],
                              lambda c:c['benchmark']==benchmark and c['model']==model)
        ledger_path = self.root/'ontology/studies/v100_roofline_537/benchmarks'/(benchmark+'.json')
        # Ledger is deliberately not pinned wholesale: unrelated model status updates
        # must not invalidate a correctly selected entry. Pin the actual selected joins.
        ledger = json.loads(ledger_path.read_bytes())
        state = ledger['models'][model]
        contract = ledger['accepted_named_contract']
        expected_manifest = capture['selected_profile_manifest']
        selected = Path(state['profile_evidence_manifest'])
        selected = selected if selected.is_absolute() else self.root/selected
        if selected.resolve() != Path(expected_manifest['path']).resolve():
            raise ValueError('Selected profile changed')
        if (contract['configuration_id'] != capture['configuration_id']
                or contract['raw_config_sha256'] != capture['raw_config_sha256']
                or contract['result_sha256'] != capture['reference_result_sha256']
                or state.get('profiling_status') != 'profile_process_completed'):
            raise ValueError('Selected configuration/result/status changed')
        self.read(selected, expected_manifest['sha256'])
        plan = self.read(selected.parent/'plan.json.gz', capture['plan_compressed_sha256'])
        prov = self.read(selected.parent/'profile_provenance.json.gz', capture['provenance_compressed_sha256'])
        if (plan['configuration_id'] != capture['configuration_id']
                or plan['raw_config_sha256'] != capture['raw_config_sha256']
                or plan['reference_result_sha256'] != capture['reference_result_sha256']):
            raise ValueError('Plan/contract join mismatch')
        result_path = Path(plan['reference_result'])
        result_path = result_path if result_path.is_absolute() else self.root/result_path
        self.read(result_path, capture['reference_result_sha256'])
        bound = {str(Path(path).resolve()):digest for path,digest in
                 {**plan.get('execution_bindings',{}), **prov.get('additional_current_profile_input_bindings',{})}.items()}
        if bound != capture['source_bindings']:
            raise ValueError('Actual selected source bindings changed')
        metadata = self.unique(overlay['implementation_metadata_bindings'],
                               lambda c:c['benchmark']==benchmark and c['model']==model)
        record = self.read(self.root/metadata['path'], metadata['sha256'])
        if record['benchmark_id'] != benchmark or record['implementation_model'] != model:
            raise ValueError('Implementation metadata identity mismatch')
        configuration = self.unique(record.get('configurations',[]),
                                    lambda c:c.get('configuration_id')==capture['configuration_id'])
        if configuration.get('result_sha256') != capture['reference_result_sha256']:
            raise ValueError('Metadata configuration/result join mismatch')
        proof = capture['adopted_stage_proof']
        packet_ref = proof['packet']
        adopted = self.read(self.root/overlay['root_adoption']['path'], overlay['root_adoption']['sha256'])
        if packet_ref not in adopted['input_packets']:
            raise ValueError('Proof packet not in adopted root receipt')
        packet = self.read(packet_ref['path'],packet_ref['sha256'])
        patch = self.unique(packet['patches'], lambda p:p['record_key']==proof['patch_record_key'])
        digest = sha(json.dumps(patch,sort_keys=True,separators=(',',':')).encode())
        if digest != proof['patch_canonical_sha256']:
            raise ValueError('Adopted proof patch mismatch')
        actual_proofs = [entry['updates'].get('ROI_membership_proof') for entry in patch['stage_patches']
                         if entry['updates'].get('ROI_membership_proof')]
        if actual_proofs != proof['stage_proofs']:
            raise ValueError('Proof contents differ from adopted packet')
        source_map = proof_sources(actual_proofs)
        if not source_map:
            raise ValueError('No explicit source proof available')
        for path,digest in source_map.items():
            if bound.get(path) != digest or sha(Path(path).read_bytes()) != digest:
                raise ValueError('Proof source is absent/stale in selected profile: '+path)
            self.inputs[path] = digest
        return capture, record, bound

    def decorate(self, benchmark, model):
        """Return isolated display values/provenance; never mutate caller records."""
        key = (benchmark,model)
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        display = {'overrides':{}, 'aliases':{}, 'provenance':[], 'withheld':[]}
        if self.overlay is None:
            return display
        entries = [e for e in self.overlay['display_overrides'] if e['benchmark']==benchmark and e['model']==model]
        aliases = [e for e in self.overlay['domain_display_aliases'] if e['benchmark']==benchmark]
        if not entries and not aliases:
            return display
        try:
            with self.lock():
                capture,record,bound = self.capture(benchmark,model)
                norm = record.get('implementation_ontology_normalized') or {}
                for entry in entries:
                    try:
                        field = entry['display_field']
                        if field not in ALLOWED_OVERRIDES or norm.get(field) != entry['expected_underlying_value']:
                            raise ValueError('Unsupported override or expected metadata value mismatch')
                        ref = entry['underlying_record']
                        self.read(self.root/ref['path'],ref['sha256'])
                        if (entry['selected_configuration_id'] != capture['configuration_id']
                                or entry['selected_result_sha256'] != capture['reference_result_sha256']
                                or entry['selected_profile_manifest'] != capture['selected_profile_manifest']):
                            raise ValueError('Override/capture identity mismatch')
                        for source_key in entry['source_evidence_keys']:
                            evidence = self.overlay['source_evidence'][source_key]
                            path = str(Path(evidence['path']).resolve())
                            if bound.get(path) != evidence['sha256'] or sha(Path(path).read_bytes()) != evidence['sha256']:
                                raise ValueError('Override source not selected and current')
                            self.inputs[path] = evidence['sha256']
                        display['overrides'][field] = copy.deepcopy(entry['proposed_display_value'])
                        display['provenance'].append({'kind':'mapping_override','field':field,'original_field':'implementation_ontology_normalized.'+field,'source_keys':entry['source_evidence_keys'],'authority':self.authority})
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        display['withheld'].append({'field':entry.get('display_field'),'reason':str(error)})
                for alias in aliases:
                    try:
                        domain = self.read(self.root/alias['domain_path'],alias['domain_sha256'])
                        for field,entry in alias['display_values'].items():
                            original = entry['original_field'].split('.')
                            value = domain
                            for name in original:
                                value = value[name]
                            if value != entry['value']:
                                raise ValueError('Alias differs from original curated text')
                        display['aliases'] = copy.deepcopy(alias['display_values'])
                        display['provenance'].append({'kind':'curated_aliases','domain_path':alias['domain_path'],'domain_sha256':alias['domain_sha256'],'original_fields':{k:v['original_field']for k,v in alias['display_values'].items()},'authority':self.authority})
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        display['withheld'].append({'field':'domain_aliases','reason':str(error)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            display['withheld'].append({'field':'entry_capture_joins','reason':str(error)})
        if display['withheld']:
            self.issues.append({'benchmark':benchmark,'model':model,'withheld':display['withheld']})
        self.cache[key] = copy.deepcopy(display)
        return display

    def summary(self):
        return {'metadata_lock_protocol':'borrowed_verified_EX_unchanged'if self.borrowed_lock_fd is not None else 'owned_SH',
                'authority':self.authority,'issues':self.issues,'verified_input_sha256':dict(sorted(self.inputs.items())),
                'scope':'Display only; original metadata, graph scores/edges, phase, numerical outcomes and measurements unchanged.'}
