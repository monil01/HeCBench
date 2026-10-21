from pathlib import Path
import json,hashlib,difflib
R=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest()
prior=Path('/tmp/roofline-completion-20261021/roi_v3_scientific_trait_review/proposed_current537_display_overlay.json');overlay=json.loads(prior.read_text());overlay['implementation_metadata_bindings']=[]
for name in sorted({c['benchmark']for c in overlay['selected_capture_bindings']}):
 for model in ['cuda_jl','ka','jacc']:
  path=f'ontology/studies/julia_abstractions_537/implementations/{name}/{model}.json';overlay['implementation_metadata_bindings'].append({'benchmark':name,'model':model,'path':path,'sha256':sha((R/path).read_bytes())})
sub='ontology/studies/v100_roofline_537/completion_reviews/current537_display_traits';dest=O/'proposed'/sub/'overlay.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(overlay,indent=2)+'\n')
# Self identity is pinned by future root authority, not inferred from current live tool.
loader=O/'proposed/ontology/tools/current537_display_traits.py';s=loader.read_text();s=s.replace('        overlay = self.overlay\n        capture =',"        self.read(self.root/self.authority['path'], self.authority['sha256'])\n        overlay = self.read(self.root/self.authority['overlay']['path'], self.authority['overlay']['sha256'])\n        capture =",1);loader.write_text(s)
review={'schema_version':1,'status':'PENDING_ROOT_REVIEW_NOT_APPROVAL','population':537,'overlay':{'path':sub+'/overlay.json','sha256':sha(dest.read_bytes())},'loader':{'path':'ontology/tools/current537_display_traits.py','sha256':sha(loader.read_bytes())},'source_adoption':overlay['root_adoption'],'limitations':['Display-only authority; never numerical or measured phase promotion.','Root must independently review exact source and set status root_approved_current537_display_traits before hashing final root_review.json.','Final root_review SHA must be supplied explicitly to consumer CLI; no default self-approval.']};(O/'root_review.template.json').write_text(json.dumps(review,indent=2)+'\n')
files=['ontology/tools/generate_ontology_report.py','ontology/tools/generate_relationship_graph.py','ontology/tools/julia_graph/graph.js','ontology/tools/report/explorer.js'];base={f:(R/f).read_text()for f in files};p=base.copy()
def change(f,a,b):
 if a not in p[f]:raise ValueError('Missing authoring anchor '+f+' '+a[:70])
 p[f]=p[f].replace(a,b,1)
f=files[0];change(f,'from study_report_metadata import context','from study_report_metadata import context\nfrom current537_display_traits import DisplayTraits')
change(f,'def build(root, population=447):','def build(root, population=447, current537_display_review_sha256=None):')
change(f,"    ont = root/'ontology'", "    traits=DisplayTraits(root,population,current537_display_review_sha256)\n    columns=COLUMNS\n    if population==537:\n        labels={'application':'Domain / application context (curated)','method':'Method / operator (curated)',\n                'type':'Benchmark type (inferred catalogue)','motif':'Computational motifs (inferred catalogue)'}\n        display_paths={'application':'display.domain_or_application_context','method':'display.method_or_operator',\n                       'accuracy':'display.accuracy','size':'display.named_scale',\n                       'mapping':'display.work_item_mapping','geometry':'display.launch_geometry'}\n        columns=[(g,k,labels.get(k,l),display_paths.get(k,path))for g,k,l,path in COLUMNS]\n    ont = root/'ontology'")
a="        props = flatten({k:v for k,v in sources.items() if k not in ['base','domain']})"
b="""        decoration=traits.decorate(bid,model) if population==537 else None
        if population==537:
            display={'work_item_mapping':get(rec,'implementation_ontology_normalized.work_item_mapping'),
                     'launch_geometry':get(rec,'implementation_ontology_normalized.launch_geometry'),
                     'method_or_operator':get(dom,'scientific_intent.mathematical_method'),
                     'domain_or_application_context':get(dom,'scientific_intent.application_domain'),
                     'accuracy':get(dom,'scientific_intent.accuracy_requirements'),
                     'named_scale':get(dom,'scientific_intent.input_size_regime')}
            display.update(decoration['overrides'])
            for key,entry in decoration['aliases'].items():display[key]=entry['value']
            sources['display']=display
            sources['display_provenance']=decoration
        props = flatten({k:v for k,v in sources.items() if k not in ['base','domain']})"""
change(f,a,b);change(f,'        values = {key:get(sources,path) for _,key,_,path in COLUMNS}','        values = {key:get(sources,path) for _,key,_,path in columns}')
change(f,"'record_path':str(file.relative_to(root))})","'record_path':str(file.relative_to(root)),**({'display_provenance':decoration}if population==537 else {})})")
change(f,"'values':{key:get(sources,path) for _,key,_,path in COLUMNS}","'values':{key:get(sources,path) for _,key,_,path in columns}")
change(f,"'columns':[{'group':g,'key':k,'label':l,'path':p} for g,k,l,p in COLUMNS]","'columns':[{'group':g,'key':k,'label':l,'path':p} for g,k,l,p in columns]")
change(f,"    payload['roofline_source_digest']", "    if population==537:payload['current537_display_traits']=traits.summary()\n    payload['roofline_source_digest']")
change(f,"    args=parser.parse_args();root=args.root.resolve();data,meta,payload=build(root,args.population)","    parser.add_argument('--current537-display-review-sha256',help='Explicit SHA256 of root-approved current537 display authority; absent disables overlays.')\n    args=parser.parse_args();root=args.root.resolve();data,meta,payload=build(root,args.population,args.current537_display_review_sha256)")
f=files[1];change(f,'from generate_ontology_report import load_roofline_registry','from generate_ontology_report import load_roofline_registry\nfrom current537_display_traits import DisplayTraits')
change(f,'def build(root,k,population=447):','def build(root,k,population=447,current537_display_review_sha256=None):\n    traits=DisplayTraits(root,population,current537_display_review_sha256)')
change(f,"        for model in MODELS:\n            workflow=", "        if population==537:\n            for model in MODELS:\n                # Scoring uses original mapping/algorithm/method fields exclusively.\n                decoration=traits.decorate(bid,model)\n                models[model]['display_traits']=decoration\n                models[model]['display_mapping']=decoration['overrides'].get('work_item_mapping',models[model]['mapping'])\n        for model in MODELS:\n            workflow=")
change(f,"    return {'meta':meta,'nodes':nodes,'edges':edges}","    if population==537:meta['current537_display_traits']=traits.summary()\n    return {'meta':meta,'nodes':nodes,'edges':edges}")
change(f,'    args=parser.parse_args()','    parser.add_argument(\'--current537-display-review-sha256\',help=\'Explicit root-approved authority SHA; absent disables current537 display overlays.\')\n    args=parser.parse_args()')
change(f,'data=build(root,args.neighbors,args.population)','data=build(root,args.neighbors,args.population,args.current537_display_review_sha256)')
f=files[2]
change(f," const n=nodesById.get(selected),m=n.models[model()];", " const n=nodesById.get(selected),m=n.models[model()];\n const display=m.display_traits,alias=display?.aliases||{};\n const displayProof=display?.provenance?.length?`<p class=\"cap\">Current537 source-bound display decoration; original graph scoring fields unchanged. ${display.provenance.map(p=>p.kind==='curated_aliases'?link(p.domain_path,'Curated source fields'):link(p.authority?.overlay?.path,'Display source overlay')).join(' · ')} · ${link(display.provenance[0]?.authority?.path,'Root authority')}</p><details><summary>Display field provenance</summary><pre>${esc(JSON.stringify(display,null,2))}</pre></details>`:'';")
change(f,"detail([['Method',n.method],['Accuracy',n.accuracy],['Named scale',n.size]])", "detail(display?[['Method / operator (curated)',alias.method_or_operator?.value??n.method],['Domain / application context (curated)',alias.domain_or_application_context?.value??n.application],['Accuracy',alias.accuracy?.value??n.accuracy],['Named scale',alias.named_scale?.value??n.size]]:[['Method',n.method],['Accuracy',n.accuracy],['Named scale',n.size]])+displayProof")
change(f,"['Mapping',m.mapping]","['Mapping',m.display_mapping??m.mapping]")
f=files[3]
change(f,"  const props={...(STUDY.base_properties[id]||{}),...row.properties};", "  const props={...(STUDY.base_properties[id]||{}),...row.properties};\n  if(row.display_provenance?.provenance?.length){\n    const display=row.display_provenance;\n    $('recordLinks').innerHTML+=' · '+display.provenance.map(p=>`<a href=\"../../${encodeURI(p.kind==='curated_aliases'?p.domain_path:p.authority.overlay.path)}\" target=\"_blank\" rel=\"noopener\">${esc(p.kind==='curated_aliases'?'Curated field source':'Current537 display overlay')}</a>`).join(' · ')+` · <a href=\"../../${encodeURI(display.provenance[0].authority.path)}\" target=\"_blank\" rel=\"noopener\">Root display authority</a>`;\n  }")
# All edits below write private copies only.
patch=[];inventory={}
for f in files:
 for group,data in [('base',base[f]),('proposed',p[f])]:
  out=O/group/f;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(data)
 inventory[f]={'base_sha256':sha(base[f].encode()),'proposed_sha256':sha(p[f].encode())}
 patch.extend(difflib.unified_diff(base[f].splitlines(True),p[f].splitlines(True),fromfile='a/'+f,tofile='b/'+f))
for f in ['ontology/tools/current537_display_traits.py',sub+'/overlay.json']:
 text=(O/'proposed'/f).read_text();inventory[f]={'new_file':True,'proposed_sha256':sha(text.encode())};patch.extend(difflib.unified_diff([],text.splitlines(True),fromfile='/dev/null',tofile='b/'+f))
(O/'minimal.patch').write_text(''.join(patch));(O/'authoring_bindings.json').write_text(json.dumps(inventory,indent=2)+'\n');print(json.dumps({'overlay_sha256':sha(dest.read_bytes()),'loader_sha256':sha(loader.read_bytes()),'patch_sha256':sha((O/'minimal.patch').read_bytes())}))
