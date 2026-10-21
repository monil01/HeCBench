from pathlib import Path
import json,hashlib,difflib,fcntl,csv,io
O=Path(__file__).parent;R=Path('/home/imo/HeCBench_julia');sha=lambda b:hashlib.sha256(b).hexdigest()
f='tools/v100_roofline_refresh.py';p=O/'proposed'/f;s=p.read_text()
a="""            authority=study/'publication_selection_status.json'
            with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                value={'current_export_available':publication_ok,'requested_stratum':args.publication_stratum,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                if publication_ok:value['publication_selection']=json.loads((export/'campaign_snapshot.json').read_text())['publication_selection']
                temp=authority.with_suffix('.json.next');temp.write_text(json.dumps(value,indent=2)+'\\n');temp.replace(authority)
            if publication_ok:
                dest=study/'publication';dest.mkdir(exist_ok=True)
                for source in export.iterdir():
                    if source.is_file():
                        next_path=dest/(source.name+'.next');shutil.copy2(source,next_path);next_path.replace(dest/source.name)
"""
b="""            authority=study/'publication_selection_status.json'
            with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                value={'current_export_available':False,'requested_stratum':args.publication_stratum,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                if publication_ok:
                    snapshot=json.loads((export/'campaign_snapshot.json').read_text())
                    # Export read lock ended already: revalidate every bound input before publishing.
                    current=True
                    for name,binding in snapshot['input_bindings'].items():
                        path=Path(name);path=path if path.is_absolute()else root/path
                        if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:current=False;break
                    if current:
                        dest=study/'publication';dest.mkdir(exist_ok=True);files={}
                        for source in export.iterdir():
                            if source.is_file():
                                next_path=dest/(source.name+'.next');shutil.copy2(source,next_path);next_path.replace(dest/source.name)
                                files[source.name]={'path':str(dest/source.name),'sha256':hashlib.sha256((dest/source.name).read_bytes()).hexdigest()}
                        value.update(current_export_available=True,publication_selection=snapshot['publication_selection'],files=files)
                    else:value['reason']='input_changed_since_export'
                temp=authority.with_suffix('.json.next');temp.write_text(json.dumps(value,indent=2)+'\\n');temp.replace(authority)
"""
assert a in s;s=s.replace(a,b,1)
a="""        # Plot in a fresh directory. Never fall back to a previous canonical PDF.
"""
b="""        # Mark both consumers unavailable before a fresh attempt; a crash cannot bless old files.
        with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for target in (study/'figures/canonical_figure_manifest.json',study/'publication_selection_status.json'):
                target.parent.mkdir(exist_ok=True)
                tmp=target.with_suffix('.json.next')
                tmp.write_text(json.dumps({'canonical_available':False,'current_export_available':False,'reason':'refresh_pending','requested_stratum':args.publication_stratum})+'\\n');tmp.replace(target)
        # Plot in a fresh directory. Never fall back to a previous canonical PDF.
"""
assert a in s;s=s.replace(a,b,1);p.write_text(s)
# Metadata-only source snapshots, no executing/importing any proposed target.
files=sorted(str(p.relative_to(O/'proposed'))for p in(O/'proposed').rglob('*')if p.is_file());patch=[];bindings={}
for f in files:
 b=(O/'base'/f).read_bytes();q=(O/'proposed'/f).read_bytes()
 bindings[f]={'base_sha256':sha(b),'proposed_sha256':sha(q)}
 patch.extend(difflib.unified_diff(b.decode().splitlines(True),q.decode().splitlines(True),fromfile='a/'+f,tofile='b/'+f))
(O/'minimal.patch').write_text(''.join(patch));(O/'authoring_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
# Capture exact current schema/applicability evidence under a short read-only shared lock.
csv.field_size_limit(64*1024*1024)
evidence={}
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 report=(R/'ontology/tools/generate_ontology_report.py').read_text();cols=report.split('COLUMNS = [',1)[1].split('\n]',1)[0]
 evidence['current_report_column_tuple_count']=cols.count(" ('")
 evidence['current_report_sha256']=sha(report.encode());evidence['scalar_applicability_records']=[]
 for file in sorted((R/'ontology/studies/v100_roofline_537/benchmarks').glob('*.json')):
  raw=file.read_bytes();j=json.loads(raw)
  for model,row in j.get('models',{}).items():
   if row.get('scalar_roofline_applicability_review'):
    evidence['scalar_applicability_records'].append({'benchmark':j.get('benchmark',file.stem),'model':model,'ledger_sha256':sha(raw),'scalar_roofline_applicability':row.get('scalar_roofline_applicability'),'review':row['scalar_roofline_applicability_review']})
 for name in ['kernel_audit.csv','kernel_roofline_points.csv','kernel_hierarchical_points.csv','kernel_tensor_roofline_points.csv']:
  file=R/'ontology/studies/v100_roofline_537'/name
  if not file.exists():continue
  raw=file.read_bytes();reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
  evidence[name]={'sha256':sha(raw),'bytes':len(raw),'columns':reader.fieldnames,'profiler_identity_columns_present':[k for k in ('profiler_method_id','profiler_version','method_review_sha256')if k in reader.fieldnames]}
 fcntl.flock(lock,fcntl.LOCK_UN)
evidence['adopted_scalar_applicability_record_count']=len(evidence['scalar_applicability_records'])
evidence['source_rebase_stable']={f:sha((R/f).read_bytes())==v['base_sha256']for f,v in bindings.items()}
(O/'current_input_review.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({'files':len(files),'base_columns':evidence['current_report_column_tuple_count'],'adopted_applicability':len(evidence['scalar_applicability_records']),'all_bases_stable':all(evidence['source_rebase_stable'].values()),'patch_sha256':sha((O/'minimal.patch').read_bytes())}))
