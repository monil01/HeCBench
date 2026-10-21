#!/usr/bin/env python3
"""Generate the Julia-focused, evidence-linked relationship graph (offline HTML)."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from study_report_metadata import context
from generate_ontology_report import load_roofline_registry
from current537_display_traits import DisplayTraits

ROOT=Path(__file__).resolve().parents[2]
MODELS=['cuda_jl','ka','jacc']
STOP=set('a an and are as at be by for from in into is it no not of on one or per same source the this to using versus vs with without kernel gpu julia cuda model array arrays native implementation'.split())


def tokens(value):
    text=json.dumps(value) if not isinstance(value,str) else value
    return text_tokens(text)


@lru_cache(maxsize=32768)
def text_tokens(text):
    if not text or text.lower() in ['null','unknown','none','not recorded']:return set()
    text=re.sub(r'([a-z])([A-Z])',r'\1 \2',text)
    return set(re.findall(r'[a-z][a-z0-9]+',text.lower()))-STOP


def overlap(a,b):
    return len(a&b)/len(a|b) if a and b else 0.0


def precision(value):
    text=json.dumps(value or '').lower()
    return set(re.findall(r'\b(?:float|fp|int|uint|bf|bfloat)(?:8|16|32|64|128)\b',text))


def get(d,path,default=None):
    for k in path.split('.'):
        if not isinstance(d,dict):return default
        d=d.get(k)
    return default if d is None else d


def unique(values):
    return sorted({str(v) for v in values if v is not None})


def build(root,k,population=447,current537_display_review_sha256=None,metadata_lock_fd=None):
    traits=DisplayTraits(root,population,current537_display_review_sha256,metadata_lock_fd)
    study=root/f'ontology/studies/julia_abstractions_{population}'
    classfile=root/'2026-PASC-Hecbench-Julia-paper/research/revision_experiments/classes/normalized_records.json'
    classes={(r['benchmark'],r['model']):r for r in json.loads(classfile.read_text())} if classfile.exists() else {}
    roofline,roofline_meta,roofline_files=load_roofline_registry(root,population)
    profile_keys=('profiling_status','configuration_ids','device_uuid','kernel_observation_count',
                  'loaded_kernel_observation_count','roi_coverage','dram_roofline_applicability',
                  'scalar_roofline_applicability','scalar_applicability_evidence_status',
                  'geometric_regimes','diagnostic_geometric_regimes','precision_classes','causal_bottleneck','evidence_link_status',
                  'evidence_paths','registry_path','registry_sha256','positioning_evidence_scope',
                  'proved_semantic_stages','sample_phase_qualifications','geometric_regime_scope')
    def profile(bid,model):
        item=roofline.get((bid,model),{})
        return {key:item.get(key) for key in profile_keys}
    nodes=[];files=list(roofline_files)
    for file in sorted((study/'domains').glob('*.json')):
        files.append(file);domain=json.loads(file.read_text());bid=domain['benchmark_id'];intent=domain.get('scientific_intent',{});models={}
        for model in MODELS:
            path=study/'implementations'/bid/(model+'.json');files.append(path);record=json.loads(path.read_text());norm=record.get('implementation_ontology_normalized') or {};review=record.get('implementation_ontology') or {};cl=classes.get((bid,model),{});cfg=record.get('configurations',[])
            events=get(record,'failure_taxonomy.events',[])
            failures=[e for e in events if e.get('is_failure') is True and e.get('origin_model')==model]
            models[model]={
                'roofline':profile(bid,model),
                'status':record['status'],'class':cl.get('controlled_class') or norm.get('implementation_class'),
                'class_basis':'reviewed class reconciliation' if cl else 'recorded declaration; unreconciled vocabulary',
                'api':cl.get('device_api_dependence'),'host_role':cl.get('host_role'),
                'algorithm':norm.get('algorithm'),'layout':norm.get('layout'),
                'mapping':norm.get('work_item_mapping'),'mapping_fidelity':norm.get('native_to_model_mapping'),
                'storage':norm.get('storage_precision'),'compute':norm.get('compute_precision'),
                'resources':norm.get('resource_policy'),'library':norm.get('library_backend_boundary'),
                'host':norm.get('host_control_boundary'),'support':norm.get('support_boundary'),
                'environment':record.get('environment',{}),'review':record.get('implementation_review_status'),
                'profiling':review.get('profiling_status'),'record_path':str(path.relative_to(root)),
                'source':review.get('source_evidence'),'configurations':cfg,
                'failure_codes':unique(e.get('taxonomy_code') for e in failures),
                'failure_events':[{'code':e.get('taxonomy_code'),'phase':e.get('phase'),'origin':e.get('origin_model'),'cause':get(e,'raw_event.cause'),'evidence_path':e.get('evidence_path')} for e in failures],
                'all_event_codes':unique(e.get('taxonomy_code') for e in events),
                'failure_history':get(record,'failure_taxonomy.history_completeness'),
                'effort':{key:get(record,'effort.'+key) for key in ['candidate_attempts','compiler_checks','compiler_feedback_repairs','runtime_repairs','numerical_repairs']},
                'token_status':get(record,'effort.translation_tokens.status'),
                'token_total':get(record,'effort.translation_tokens.total_tokens'),
                'token_windows':get(record,'effort.translation_tokens.observed_activity_windows',[]),
                'saved_source_token_proxy':get(record,'effort.translation_tokens.estimated_saved_source_tokens'),
                'agreement':unique(c.get('numerical_agreement_kind') for c in cfg),
                'timing_scope':norm.get('timing_scope') or get(record,'shared_benchmark_traits.timing_scope'),
                'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_commit':record.get('source_commit')}
        if population==537:
            for model in MODELS:
                # Scoring uses original mapping/algorithm/method fields exclusively.
                decoration=traits.decorate(bid,model)
                models[model]['display_traits']=decoration
                models[model]['display_mapping']=decoration['overrides'].get('work_item_mapping',models[model]['mapping'])
        for model in MODELS:
            workflow=study/'workflow_studies'/bid/(model+'.json')
            if workflow.exists():models[model]['additional_workflow_studies']=json.loads(workflow.read_text());files.append(workflow)
        nodes.append({'id':bid,'native_roofline':profile(bid,'cuda'),'family':intent.get('primary_family','unknown'),'task':intent.get('scientific_task'),
                      'method':intent.get('mathematical_method'),'application':intent.get('application_domain'),
                      'accuracy':intent.get('accuracy_requirements'),'size':intent.get('input_size_regime'),
                      'intent_review':intent.get('review_status'),'domain_path':str(file.relative_to(root)),
                      'domain_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'scientific_sources':intent.get('evidence',[]),'models':models})
    edges=[]
    # Compute every pair, then retain the union of each endpoint's top-k positive links.
    for view in ['science','implementation','verification']:
        for model in (['shared'] if view=='science' else MODELS):
            candidates=[]
            for i,a in enumerate(nodes):
                for b in nodes[i+1:]:
                    evidence=[];score=0.0
                    if view=='science':
                        same=a['family']==b['family'] and a['family'] not in ['unknown','unclassified']
                        task=overlap(tokens(a['task']),tokens(b['task']));method=overlap(tokens(a['method']),tokens(b['method']))
                        if not same and not(task>=.3 and method>=.3):continue
                        if same:evidence.append({'property':'scientific_intent.primary_family','shared':[a['family']],'weight':.45,'contribution':.45})
                        for prop,sim,ta,tb,weight in [('scientific_intent.scientific_task',task,a['task'],b['task'],.25),('scientific_intent.mathematical_method',method,a['method'],b['method'],.30)]:
                            if sim:evidence.append({'property':prop,'shared':sorted(tokens(ta)&tokens(tb)),'weight':weight,'contribution':round(weight*sim,6),'comparison':'token Jaccard; lexical evidence, not semantic equivalence'})
                        score=(.45 if same else 0)+.25*task+.30*method
                    else:
                        aa=a['models'][model];bb=b['models'][model]
                        if view=='implementation':
                            if not aa['configurations'] or not bb['configurations']:continue
                            if not aa['class'] or aa['class']!=bb['class'] or aa['class'] in ['unavailable','unknown','multiple_config_classes']:continue
                            sims={field:overlap(tokens(aa[field]),tokens(bb[field])) for field in ['algorithm','layout','mapping']}
                            if max(sims.values())<.08:continue
                            evidence.append({'property':'reviewed implementation class','shared':[aa['class']],'weight':.30,'contribution':.30})
                            score=.30
                            if aa['api'] and aa['api']==bb['api']:
                                score+=.10;evidence.append({'property':'device API dependence','shared':[aa['api']],'weight':.10,'contribution':.10})
                            for field,weight in [('algorithm',.20),('layout',.15),('mapping',.10)]:
                                sim=sims[field]
                                if sim:evidence.append({'property':field,'shared':sorted(tokens(aa[field])&tokens(bb[field])),'weight':weight,'contribution':round(weight*sim,6),'comparison':'token Jaccard; lexical evidence'})
                                score+=weight*sim
                            pa,pb=precision(aa['storage']),precision(bb['storage']);sim=overlap(pa,pb);score+=.15*sim
                            if sim:evidence.append({'property':'storage precision','shared':sorted(pa&pb),'weight':.15,'contribution':round(.15*sim,6)})
                        else:
                            fa,fb=set(aa['failure_codes']),set(bb['failure_codes']);common=fa&fb
                            kinds=set(aa['agreement'])&set(bb['agreement'])
                            qualified=aa['status']==bb['status'] and aa['status'] not in ['verified_named_configuration']
                            special=any('property' in x or 'execution' in x for x in kinds)
                            if not common and not qualified and not special:continue
                            sim=overlap(fa,fb);score=.5*sim
                            if common:evidence.append({'property':'actual failure codes with matching origin model','shared':sorted(common),'weight':.50,'contribution':round(.5*sim,6)})
                            sim=overlap(set(aa['agreement']),set(bb['agreement']));score+=.30*sim
                            if sim:evidence.append({'property':'recorded agreement kind','shared':sorted(kinds),'weight':.30,'contribution':round(.3*sim,6),'comparison':'shared descriptor; does not make contracts interchangeable'})
                            if qualified:score+=.20;evidence.append({'property':'qualified outcome','shared':[aa['status']],'weight':.20,'contribution':.20})
                    if score>0:candidates.append({'source':a['id'],'target':b['id'],'view':view,'model':model,'score':round(score,6),'evidence':evidence})
            neighbors={n['id']:[] for n in nodes}
            for idx,e in enumerate(candidates):
                neighbors[e['source']].append((idx,e));neighbors[e['target']].append((idx,e))
            keep=set()
            for nid,items in neighbors.items():
                items.sort(key=lambda p:(-p[1]['score'],p[1]['target'] if p[1]['source']==nid else p[1]['source']))
                keep.update(i for i,_ in items[:k])
            edges.extend(candidates[i] for i in sorted(keep))
    if classfile.exists():files.append(classfile)
    meta={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'assessed_names':len(nodes),'implementation_records':len(nodes)*3,'top_k_per_endpoint':k,
          'population_context':context(root,population),'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest(),
          'study_path':str(study.relative_to(root)),'class_source':str(classfile.relative_to(root)) if classfile.exists() else None,
          'class_sha256':hashlib.sha256(classfile.read_bytes()).hexdigest() if classfile.exists() else None,
          'edge_counts':dict(Counter(e['view']+':'+e['model'] for e in edges)),
          'method':'All positive pairwise candidates, then undirected union of endpoint top-k; ties by neighbor ID. Scores are fixed-weight descriptive feature matches, not learned embeddings, probabilities, performance predictions or causal effects. No missing value is imputed. Original catalogue similarity_edges.jsonl is preserved.'}
    meta['roofline_registry_summary']={k:v for k,v in roofline_meta.items() if k!='linked_evidence_sha256'}
    meta['roofline_linked_evidence_digest']=hashlib.sha256(json.dumps(roofline_meta.get('linked_evidence_sha256',{}),sort_keys=True).encode()).hexdigest()
    meta['roofline_interpretation']='Configuration-specific partial per-kernel geometric projections, never whole-workflow or causal labels; profile filters do not change descriptive edge weights.'
    if population==537:meta['current537_display_traits']=traits.summary()
    return {'meta':meta,'nodes':nodes,'edges':edges}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path);parser.add_argument('--neighbors',type=int,default=5);parser.add_argument('--population',type=int,choices=[447,537],default=447)
    parser.add_argument('--current537-display-review-sha256',help='Explicit root-approved authority SHA; absent disables current537 display overlays.')
    parser.add_argument('--metadata-lock-fd',type=int,help='Borrowed canonical metadata EX-lock FD inherited via pass_fds; never downgraded, unlocked or closed.')
    args=parser.parse_args()
    if not 1<=args.neighbors<=30:parser.error('--neighbors must be between 1 and 30')
    root=args.root.resolve();data=build(root,args.neighbors,args.population,args.current537_display_review_sha256,args.metadata_lock_fd);assets=Path(__file__).parent/'julia_graph'
    template=(assets/'template.html').read_text().replace('__TOP_K__',str(args.neighbors)).replace('__GRAPH_CSS__',(assets/'graph.css').read_text()).replace('__GRAPH_JS__',(assets/'graph.js').read_text()).replace('__GRAPH_DATA__',json.dumps(data,separators=(',',':')).replace('<','\\u003c'))
    template=template.replace('__STUDY_SUMMARY__',context(root,args.population)['summary']).replace('__STUDY_POPULATION__',str(args.population))
    out=args.output or root/'ontology/reports/relationship_graph.html';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(template)
    index=root/'ontology/relationships/julia_relationships.json';index.write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(json.dumps({'output':str(out),'index':str(index),'nodes':len(data['nodes']),'edges':len(data['edges']),'bytes':out.stat().st_size,'edge_counts':data['meta']['edge_counts']}))


if __name__=='__main__':main()
