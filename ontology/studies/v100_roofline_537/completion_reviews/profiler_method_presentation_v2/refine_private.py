from pathlib import Path
import hashlib,json,difflib
OUT=Path(__file__).parent; ROOT=Path('/home/imo/HeCBench_julia')
selection='''
def stratum_id(record):
    return hashlib.sha256(json.dumps(profiler_stratum(record),ensure_ascii=True,separators=(',',':')).encode()).hexdigest()

def publication_choice(records, requested):
    choices={stratum_id(record):profiler_identity(record) for record in records}
    if requested is not None:
        if requested not in choices:
            raise ValueError('Requested publication stratum absent; choose exact SHA256 from '+json.dumps(choices,sort_keys=True))
        chosen=requested
    elif len(choices)==1:
        chosen=next(iter(choices))
    elif not choices:
        chosen=None
    else:
        raise ValueError('Multiple profiler strata require --publication-stratum EXACT_SHA256; no canonical export: '+json.dumps(choices,sort_keys=True))
    return {'stratum_id':chosen,'identity':choices.get(chosen),'selection':'explicit' if requested else 'unique' if chosen else 'no_compatible_points',
            'available_strata':choices,'identity_is_calibration_approval':False}
'''
def read(f):return (OUT/'proposed'/f).read_text()
def save(f,s):
 p=OUT/'proposed'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def change(s,a,b):
 if a not in s:raise ValueError('missing '+a[:100])
 return s.replace(a,b,1)
f='tools/v100_roofline_plot.py';s=read(f)
s=change(s,'import re\n','import re\nimport shutil\nimport textwrap\n')
s=change(s,'def select_points(rows):',selection+'\ndef compact_profiler_version(raw):\n    # Display only: exact unabridged bytes remain in profiler_strata metadata.\n    match=re.search(r"(?i)(?:version|release)\\s*[:=]?\\s*(\\d+(?:\\.\\d+){1,4})",raw)\n    label="NCU "+match.group(1) if match else " ".join(raw.split())\n    return textwrap.shorten(label,width=60,placeholder="…")\n\ndef select_points(rows):')
s=change(s,"    fig.suptitle('V100 kernel observations — normalized precision rooflines\\n'+identity['profiler_method_id']+' | '+identity['profiler_version'], y=.995, fontsize=10)","    fig.suptitle('V100 kernel observations — normalized precision rooflines', y=.99, fontsize=12)\n    label=textwrap.shorten(identity['profiler_method_id'],width=64,placeholder='…')+' | '+compact_profiler_version(identity['profiler_version'])\n    fig.text(.5,.958,label+' | stratum '+stratum_id(identity)[:12],ha='center',fontsize=8)")
s=change(s,"        path = output_prefix.with_suffix(suffix)","        path = Path(str(output_prefix)+suffix)  # Append: with_suffix would erase .method_TAG.")
s=change(s,"    args = parser.parse_args()","    parser.add_argument('--publication-stratum',help='Exact SHA256 of method/version/review tuple; required if multiple compatible strata exist.')\n    args = parser.parse_args()")
s=change(s,"    points, exclusions = select_points(rows)","    points, exclusions = select_points(rows)\n    # Gate before writing figures; no fallback to sorted-first or a stale canonical file.\n    publication=publication_choice(points,args.publication_stratum)")
s=change(s,"        tag=sha256_bytes(json.dumps(key).encode())[:12]","        tag=stratum_id(subset[0])")
s=change(s,"                           tag=tag,plotted_rows=len(subset),outputs=[str(path)for path in paths]))","                           stratum_id=tag,plotted_rows=len(subset),outputs=[str(path)for path in paths]))\n    canonical_outputs=[]\n    if publication['stratum_id']:\n        selected=next(item for item in strata if item['stratum_id']==publication['stratum_id'])\n        for value in selected['outputs']:\n            source=Path(value);target=Path(str(prefix)+source.suffix)\n            shutil.copyfile(source,target);canonical_outputs.append(target)")
s=change(s,"        'profiler_strata':strata,","        'profiler_strata':strata,\n        'publication_selection':publication,\n        'canonical_outputs':[{'path':str(p.resolve()),'sha256':sha256_bytes(p.read_bytes())}for p in canonical_outputs],\n        'canonical_consumer_contract':'Consume only via the current metadata publication_selection and canonical_outputs SHA bindings, with matching current CSV input hashes; never use an unbound existing PDF. Ambiguous selection raises before output; refresh uses fresh temporary output directory.',")
s=change(s,"    print(json.dumps({'metadata': str(destination), 'counts': metadata['counts']}))","    print(json.dumps({'metadata': str(destination), 'counts': metadata['counts'],'publication_selection':publication}))")
save(f,s)
f='tools/export_v100_roofline_paper.py';s=read(f)
s=change(s,'import argparse,collections,contextlib,csv,fcntl,hashlib,io,json,math,pathlib,sys','import argparse,collections,contextlib,csv,fcntl,hashlib,io,json,math,pathlib,sys')
s=change(s,'def counter_dict(c):',selection+'\ndef counter_dict(c):')
s=change(s,"ap.add_argument('--usage-receipt',type=P);","ap.add_argument('--usage-receipt',type=P);ap.add_argument('--publication-stratum',help='Exact SHA256 method/version/review identity; required for multiple completed projection strata');")
s=change(s," stats={m:"," # Canonical geometric publication is a single explicitly selected (or unique) stratum.\n positioned=[r for r in hier_rows if r.get('profiling_status')==COMPLETE and ledgers.get(r['benchmark'],{}).get('models',{}).get(r['model'],{}).get('profiling_status')==COMPLETE and not (r.get('dram_roofline_applicability')or'').startswith('not_applicable') and (num(r.get('executed_flops'))or 0)>0 and r.get('roofline_regime')in REGIMES and num(r.get('intensity_over_ridge'))is not None]\n publication=publication_choice(positioned,a.publication_stratum)\n stats={m:")
s=change(s,"  family=familyrecords[r['benchmark']]['family'];predictions", "  if stratum_id(r)!=publication['stratum_id']:excluded['different_publication_profiler_stratum']+=1;continue\n  family=familyrecords[r['benchmark']]['family'];predictions")
s=change(s,"payload={'schema_version':1,", "payload={'publication_selection':publication,'coverage_counts_span_all_strata':True,'positioned_prediction_counts_single_stratum':True,'schema_version':1,")
save(f,s)
f='tools/v100_roofline_refresh.py';s=(ROOT/f).read_text();(OUT/'base'/f).write_text(s)
s=change(s,"    ap.add_argument('--once',action='store_true')","    ap.add_argument('--once',action='store_true')\n    ap.add_argument('--publication-stratum',help='Exact method/version/review SHA256; mandatory for multi-stratum canonical figures/publication predictions.')")
s=change(s,"        run('roofline_plot',['/home/imo/noback/conda/bin/python3.9',str(root/'tools/v100_roofline_plot.py')])",'''        # Plot in a fresh directory. Never fall back to a previous canonical PDF.
        # A manifest is the sole canonical authority; old files alone are not current evidence.
        with tempfile.TemporaryDirectory(prefix='roofline-figures-',dir='/tmp')as workspace:
            command=['/home/imo/noback/conda/bin/python3.9',str(root/'tools/v100_roofline_plot.py'),'--output-dir',workspace]
            if args.publication_stratum:command+=['--publication-stratum',args.publication_stratum]
            ok=run('roofline_plot',command)
            with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                dest=study/'figures';dest.mkdir(exist_ok=True)
                manifest={'canonical_available':False,'reason':'fresh_plot_failed_or_ambiguous','requested_stratum':args.publication_stratum}
                if ok:
                    meta=json.loads((Path(workspace)/'normalized_kernel_rooflines.metadata.json').read_text())
                    source_hash=hashlib.sha256((study/'kernel_roofline_points.csv').read_bytes()).hexdigest()
                    tensor=study/'kernel_tensor_roofline_points.csv'
                    tensor_hash=hashlib.sha256(tensor.read_bytes()).hexdigest()if tensor.is_file()else None
                    if (source_hash==meta['input']['sha256'] and tensor_hash==(meta['tensor_input']or{}).get('sha256') and meta['canonical_outputs']):
                        # Rebase paths in the metadata; retain all full raw identity strings.
                        for group in ('outputs','canonical_outputs'):
                            for item in meta[group]:item['path']=str(dest/Path(item['path']).name)
                        for item in meta['profiler_strata']:item['outputs']=[str(dest/Path(p).name)for p in item['outputs']]
                        for key in ('input','tensor_input'):
                            if meta.get(key):meta[key]['snapshot']=str(dest/Path(meta[key]['snapshot']).name)
                        for source in Path(workspace).iterdir():
                            if source.name.endswith('.metadata.json'):continue
                            target=dest/(source.name+'.next');shutil.copy2(source,target);target.replace(dest/source.name)
                        metadata=dest/'normalized_kernel_rooflines.metadata.json'
                        tmp=metadata.with_suffix('.json.next');tmp.write_text(json.dumps(meta,indent=2)+'\\n');tmp.replace(metadata)
                        manifest={'canonical_available':True,'publication_selection':meta['publication_selection'],
                                  'metadata':{'path':str(metadata),'sha256':hashlib.sha256(metadata.read_bytes()).hexdigest()},
                                  'input_sha256':source_hash,'tensor_input_sha256':tensor_hash,'canonical_outputs':meta['canonical_outputs']}
                    else:manifest['reason']='input_changed_since_plot_or_no_compatible_points'
                manifest['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                target=dest/'canonical_figure_manifest.json';tmp=target.with_suffix('.json.next');tmp.write_text(json.dumps(manifest,indent=2)+'\\n');tmp.replace(target)
''')
s=change(s,"            receipt=study/'publication_export_usage.json'","            if args.publication_stratum:command+=['--publication-stratum',args.publication_stratum]\n            receipt=study/'publication_export_usage.json'")
s=change(s,"            if run('publication_snapshot',command):", "            publication_ok=run('publication_snapshot',command)\n            # Invalidate publication authority on any failed/ambiguous fresh export; old files are history only.\n            authority=study/'publication_selection_status.json'\n            with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:\n                fcntl.flock(lock,fcntl.LOCK_EX)\n                value={'current_export_available':publication_ok,'requested_stratum':args.publication_stratum,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}\n                if publication_ok:value['publication_selection']=json.loads((export/'campaign_snapshot.json').read_text())['publication_selection']\n                temp=authority.with_suffix('.json.next');temp.write_text(json.dumps(value,indent=2)+'\\n');temp.replace(authority)\n            if publication_ok:")
save(f,s)
# New current-source base snapshots; private generation only, never imports candidate tools.
files=sorted(str(p.relative_to(OUT/'proposed'))for p in(OUT/'proposed').rglob('*')if p.is_file());patch=[];bindings={}
for f in files:
 b=(OUT/'base'/f).read_text();p=read(f)
 bindings[f]={'base_sha256':hashlib.sha256(b.encode()).hexdigest(),'proposed_sha256':hashlib.sha256(p.encode()).hexdigest()}
 patch.extend(difflib.unified_diff(b.splitlines(True),p.splitlines(True),fromfile='a/'+f,tofile='b/'+f))
(OUT/'minimal.patch').write_text(''.join(patch));(OUT/'authoring_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
