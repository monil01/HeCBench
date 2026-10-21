from pathlib import Path
import hashlib,json,difflib
ROOT=Path('/home/imo/HeCBench_julia');OUT=Path('/tmp/roofline-completion-20261021/profiler_method_presentation_patch')
files=['tools/v100_roofline_plot.py','tools/export_v100_roofline_paper.py','ontology/tools/generate_ontology_report.py','ontology/tools/generate_relationship_graph.py','ontology/tools/julia_graph/graph.js','ontology/tools/julia_graph/template.html']
base={f:(ROOT/f).read_text() for f in files}; p=base.copy()
helper='''\ndef profiler_identity(record):
    """Descriptive identity only; never promote unknown identity to calibration approval."""
    declared=record.get('profiler_method') or {}
    if not isinstance(declared,dict): declared={}
    method=record.get('profiler_method_id') or declared.get('method_id')
    if not method or method=='legacy2022_recorded_or_identity_incomplete':
        method='unknown_profiler_method'
    return {'profiler_method_id':method,
            'profiler_version':record.get('profiler_version') or 'unknown_profiler_version',
            'method_review_sha256':record.get('method_review_sha256') or declared.get('review_sha256') or 'unknown_method_review'}
def profiler_stratum(record):
    identity=profiler_identity(record)
    return tuple(identity[k] for k in ('profiler_method_id','profiler_version','method_review_sha256'))
'''
def replace(f,a,b):
    if a not in p[f]:raise ValueError('Authoring anchor missing: '+f+' '+a[:100])
    p[f]=p[f].replace(a,b,1)
f=files[0]
replace(f,'def select_points(rows):',helper+'\ndef select_points(rows):')
replace(f,'        if reason:\n            exclusions.append', '        identity.update(profiler_identity(row))\n        if reason:\n            exclusions.append')
replace(f,"    outputs, plot_limits = draw(points, prefix)",'''    # Never silently overlay profiler methods. Missing historical identity is its own stratum.
    outputs=[]; plot_limits={}; strata=[]
    for key in sorted({profiler_stratum(point) for point in points}):
        subset=[point for point in points if profiler_stratum(point)==key]
        tag=sha256_bytes(json.dumps(key).encode())[:12]
        method_prefix=prefix.with_name(prefix.name+'.method_'+tag)
        paths,bounds=draw(subset,method_prefix)
        outputs.extend(paths); plot_limits[tag]=bounds
        strata.append(dict(zip(('profiler_method_id','profiler_version','method_review_sha256'),key),
                           tag=tag,plotted_rows=len(subset),outputs=[str(path)for path in paths]))''')
replace(f,"    fig.suptitle('V100 kernel observations — normalized precision rooflines', y=.98, fontsize=13)","    identity=profiler_identity(points[0]) if points else profiler_identity({})\n    fig.suptitle('V100 kernel observations — normalized precision rooflines\\n'+identity['profiler_method_id']+' | '+identity['profiler_version'], y=.995, fontsize=10)")
replace(f,"        'tensor_input':tensor_input,","        'tensor_input':tensor_input,\n        'profiler_strata':strata,\n        'profiler_policy':'Separate figures per method/version/review; unknown identity remains unknown, not assumed legacy2022.',")
f=files[1];replace(f,'def counter_dict(c):',helper+'\ndef counter_dict(c):')
replace(f,"'DRAM_tier_applicability_counts':collections.Counter()", "'profiler_stratum_completed_sample_counts':collections.Counter(),'DRAM_tier_applicability_counts':collections.Counter()")
replace(f,"   slotrows.append({'benchmark':name", "   slotrows.append({**profiler_identity(state),'benchmark':name")
replace(f,"  st=stats[m];st['completed_kernel_observations']+=1;", "  st=stats[m];st['profiler_stratum_completed_sample_counts'][json.dumps(profiler_stratum(r))]+=1;st['completed_kernel_observations']+=1;")
replace(f,"predictions[(m,r.get('precision')", "predictions[(*profiler_stratum(r),m,r.get('precision')")
replace(f,"   if not(native.get('full_timed_ROI_coverage_confirmed')", "   ns,ts=profiler_stratum(native),profiler_stratum(other)\n   if any(value.startswith('unknown_')for value in ns+ts):rejection['profiler_method_identity_incomplete']+=1;continue\n   if ns!=ts:rejection['profiler_method_version_or_review_mismatch']+=1;continue\n   if not(native.get('full_timed_ROI_coverage_confirmed')")
replace(f,"pairs.append({'benchmark':name", "pairs.append({**profiler_identity(other),'benchmark':name")
replace(f,"predrows=[{'model':m", "predrows=[{'profiler_method_id':method,'profiler_version':version,'method_review_sha256':review,'model':m")
replace(f,"for(m,p,tier,regime,ph,family,applicability),n in sorted(predictions.items())]", "for(method,version,review,m,p,tier,regime,ph,family,applicability),n in sorted(predictions.items())]")
replace(f,"predrows,['model','precision'","predrows,['profiler_method_id','profiler_version','method_review_sha256','model','precision'")
# Ontology columns remain additive to the actual current root version, never replace its live source.
f=files[2];replace(f,'def load_roofline_registry(root, population):',helper+'\ndef load_roofline_registry(root, population):')
replace(f,"            summary = {'profiling_status':status", "            summary = {**profiler_identity(entry),'profiling_status':status")
anchor=" ('V100 roofline observations','v100_status'"
pos=p[f].index(anchor)
p[f]=p[f][:pos]+" ('V100 roofline observations','v100_method','V100 profiler method','roofline.profiler_method_id'),\n ('V100 roofline observations','v100_version','V100 profiler version','roofline.profiler_version'),\n ('V100 roofline observations','v100_method_review','V100 method review SHA','roofline.method_review_sha256'),\n"+p[f][pos:]
anchor=" ('Native CUDA roofline baseline','v100_native_status'";pos=p[f].index(anchor)
p[f]=p[f][:pos]+" ('Native CUDA roofline baseline','v100_native_method','Native CUDA profiler method','roofline_native.profiler_method_id'),\n ('Native CUDA roofline baseline','v100_native_version','Native CUDA profiler version','roofline_native.profiler_version'),\n ('Native CUDA roofline baseline','v100_native_method_review','Native CUDA method review SHA','roofline_native.method_review_sha256'),\n"+p[f][pos:]
f=files[3];replace(f,"profile_keys=('profiling_status'", "profile_keys=('profiler_method_id','profiler_version','method_review_sha256','profiling_status'")
f=files[4]
replace(f," options('profileRegime',", " options('profileMethod',[...new Set(DATA.nodes.map(n=>n.models[model()].roofline?.profiler_method_id||'unknown_profiler_method'))].sort(),'profiler methods');\n options('profileVersion',[...new Set(DATA.nodes.map(n=>n.models[model()].roofline?.profiler_version||'unknown_profiler_version'))].sort(),'profiler versions');\n options('profileRegime',")
replace(f,"&&(!$('profileRegime').value", "&&(!$('profileMethod').value||(m.roofline?.profiler_method_id||'unknown_profiler_method')===$('profileMethod').value)&&(!$('profileVersion').value||(m.roofline?.profiler_version||'unknown_profiler_version')===$('profileVersion').value)&&(!$('profileRegime').value")
p[f]=p[f].replace("'profileStatus','profileRegime'","'profileStatus','profileMethod','profileVersion','profileRegime'")
replace(f,"['V100 profile status',m.roofline?.profiling_status]", "['V100 profiler method',m.roofline?.profiler_method_id],['V100 profiler version',m.roofline?.profiler_version],['V100 method review SHA',m.roofline?.method_review_sha256],['V100 profile status',m.roofline?.profiling_status]")
f=files[5];replace(f,'<label>Sampled kernel geometry', '<label>V100 profiler method<select id="profileMethod"></select></label>\n<label>V100 profiler version<select id="profileVersion"></select></label>\n<label>Sampled kernel geometry')
patch=[];inventory={}
for f in files:
    for directory,data in [('base',base[f]),('proposed',p[f])]:
        dest=OUT/directory/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(data)
    inventory[f]={'base_sha256':hashlib.sha256(base[f].encode()).hexdigest(),'proposed_sha256':hashlib.sha256(p[f].encode()).hexdigest()}
    patch.extend(difflib.unified_diff(base[f].splitlines(True),p[f].splitlines(True),fromfile='a/'+f,tofile='b/'+f))
(OUT/'minimal.patch').write_text(''.join(patch));(OUT/'authoring_bindings.json').write_text(json.dumps(inventory,indent=2)+'\n')
