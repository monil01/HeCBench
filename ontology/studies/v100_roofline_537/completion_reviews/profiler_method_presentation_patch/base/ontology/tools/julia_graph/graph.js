const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const text=v=>v===null||v===undefined||(Array.isArray(v)&&!v.length)?'Unknown':typeof v==='object'?JSON.stringify(v):String(v);
const LABELS={cuda_jl:'CUDA.jl',ka:'KernelAbstractions',jacc:'JACC',cuda:'Native CUDA reference'};
const STATUSES={verified_named_configuration:'Ordinary named contract',conditionally_verified_native_reference_repaired:'Corrected native reference',verified_native_behavior_known_scientific_defect:'Known native defect',conditionally_verified_revised_acceptance_contract:'Revised acceptance',execution_only_verified:'Execution only',hardware_blocked:'Native hardware barrier'};
const MODEL_COLORS={cuda:'#65727c',cuda_jl:'#2977c9',ka:'#199d72',jacc:'#df7328'};
const PALETTE=['#2977c9','#199d72','#df7328','#9061b7','#c15679','#478a95','#a28b38','#5974a6','#8a6f58','#7d8c40','#785f9c','#6d7f8b'];
const CLASS_COLORS={framework_device:'#199d72',cuda_device_extension:'#df7328',host_control_pipeline:'#9061b7',kernel_vendor_pipeline:'#2977c9',vendor_runtime_only:'#478a95',multiple_config_classes:'#a28b38',unavailable:'#8a6f58',unknown:'#9aa3a9'};
const STATUS_COLORS={verified_named_configuration:'#199d72',conditionally_verified_native_reference_repaired:'#2977c9',verified_native_behavior_known_scientific_defect:'#df7328',conditionally_verified_revised_acceptance_contract:'#9061b7',execution_only_verified:'#478a95',hardware_blocked:'#9b7a4e'};
const nodesById=new Map(DATA.nodes.map(n=>[n.id,n]));
const families=[...new Set(DATA.nodes.map(n=>n.family))].sort();
const familyColor=new Map(families.map((f,i)=>[f,PALETTE[i%PALETTE.length]]));
const canvas=$('graph'),ctx=canvas.getContext('2d');
let selected=nodesById.has('heat')?'heat':DATA.nodes[0].id,visibleNodes=[],visibleEdges=[],positions=new Map(),selectedEdge=null;
let scale=1,pan={x:0,y:0},width=800,height=550,pointer=null,moved=false;
const model=()=>$('model').value;
const view=()=>$('view').value;
const displayLabel=n=>n.model?LABELS[n.model]:n.id;
function options(id,values,label){const old=$(id).value;$(id).innerHTML=`<option value="">All ${label}</option>`+values.map(v=>`<option value="${esc(v)}">${esc(STATUSES[v]||v.replaceAll('_',' '))}</option>`).join('');if(values.includes(old))$(id).value=old;}
function populateFilters(){
 options('family',families,'scientific families');
 options('status',[...new Set(DATA.nodes.map(n=>n.models[model()].status))].sort(),'outcomes');
 options('implClass',[...new Set(DATA.nodes.map(n=>n.models[model()].class).filter(Boolean))].sort(),'implementation classes');
 options('failure',[...new Set(DATA.nodes.flatMap(n=>n.models[model()].failure_codes))].sort(),'actual failure codes');
 options('profileStatus',[...new Set(DATA.nodes.map(n=>n.models[model()].roofline?.profiling_status).filter(Boolean))].sort(),'V100 profile statuses');
 options('profileRegime',[...new Set(DATA.nodes.flatMap(n=>n.models[model()].roofline?.geometric_regimes||[]))].sort(),'sampled geometric regimes');
}
function benchmarkChoices(){
 const q=$('search').value.toLowerCase().trim();const rank=n=>n.id.toLowerCase()===q?0:n.id.toLowerCase().startsWith(q)?1:n.id.toLowerCase().includes(q)?2:3;const matches=DATA.nodes.filter(n=>!q||[n.id,n.task,n.application,n.models[model()].algorithm].join(' ').toLowerCase().includes(q)).sort((a,b)=>rank(a)-rank(b)||a.id.localeCompare(b.id));
 $('benchmark').innerHTML=matches.map(n=>`<option value="${esc(n.id)}">${esc(n.id)}</option>`).join('')||'<option value="">No matching benchmark</option>';
 if(q&&matches.some(n=>n.id.toLowerCase()===q)){selected=matches.find(n=>n.id.toLowerCase()===q).id;$('benchmark').value=selected;}
 else if(matches.some(n=>n.id===selected))$('benchmark').value=selected;
 else if(matches.length)selected=matches[0].id;
}
function eligible(n){
 const m=n.models[model()];return(!$('family').value||n.family===$('family').value)&&(!$('status').value||m.status===$('status').value)&&(!$('implClass').value||m.class===$('implClass').value)&&(!$('failure').value||m.failure_codes.includes($('failure').value))&&(!$('profileStatus').value||m.roofline?.profiling_status===$('profileStatus').value)&&(!$('profileRegime').value||(m.roofline?.geometric_regimes||[]).includes($('profileRegime').value))&&(!$('precision').value||JSON.stringify(m.storage??'').toLowerCase().includes($('precision').value.toLowerCase().trim()));
}
function showGraph(){
 const translation=view()==='translation';
 for(const id of ['scope','family','status','implClass','precision','failure','profileStatus','profileRegime','score','neighbors'])$(id).disabled=translation;
 $('scoreLabel').textContent=Number($('score').value).toFixed(2);$('neighborLabel').textContent=$('neighbors').value;
 selectedEdge=null;clearEvidence();
 if(translation){
  const n=nodesById.get(selected);visibleNodes=['cuda','cuda_jl','ka','jacc'].map(m=>({id:selected+'::'+m,baseId:selected,model:m,family:n.family}));
  visibleEdges=['cuda_jl','ka','jacc'].map(m=>({source:selected+'::cuda',target:selected+'::'+m,view:'translation',model:m,score:null,status:n.models[m].status,evidence:[{property:'same benchmark implementation record',shared:[selected],comparison:'Reference/implementation identity; agreement is qualified by the exact saved configuration contract.'},{property:'recorded agreement kinds',shared:n.models[m].agreement},{property:'scoped outcome',shared:[n.models[m].status]}]}));
  $('graphNote').textContent='Three recorded Julia versions of the selected benchmark. Dashed links indicate a hardware barrier. Inspect each configuration for numerical, property, execution-only or corrected-reference scope; links do not imply universal equality.';
 }else{
  const allowed=DATA.nodes.filter(eligible),ids=new Set(allowed.map(n=>n.id));
  const edges=DATA.edges.filter(e=>e.view===view()&&(e.model==='shared'||e.model===model())&&e.score>=Number($('score').value)&&ids.has(e.source)&&ids.has(e.target));
  if($('scope').value==='neighborhood'){
   const near=edges.filter(e=>e.source===selected||e.target===selected).sort((a,b)=>b.score-a.score||(a.source+a.target).localeCompare(b.source+b.target)).slice(0,Number($('neighbors').value));
   const keep=new Set(ids.has(selected)?[selected]:[]);near.forEach(e=>{keep.add(e.source);keep.add(e.target);});
   visibleNodes=allowed.filter(n=>keep.has(n.id));visibleEdges=edges.filter(e=>keep.has(e.source)&&keep.has(e.target));
  }else{visibleNodes=allowed;visibleEdges=edges;}
  $('graphNote').textContent=(view()==='science'?'Scientific links use reviewed intent; node colors show scientific family. ':view()==='implementation'?'Links compare properties within '+LABELS[model()]+'. Node colors show reviewed implementation class. ':'Links use actual origin-matched failures and qualified verification descriptors. Node colors show scoped outcome. ')+($('scope').value==='all'?'Isolated names remain visible. Whole-study labels show the selected node; zoom or choose a neighborhood for detail. ':'The neighborhood includes the selected benchmark and its strongest retained eligible connections. ')+(visibleNodes.length?'':'No nodes match the current filters; clear filters or choose another benchmark. ');
 }
 $('visibleCount').textContent=`${visibleNodes.length} nodes / ${visibleEdges.length} connections`;
 placeNodes();fitGraph();renderSelection();renderEdges();renderLegend();
}
function placeNodes(){
 positions=new Map();
 if(view()==='translation'){
  positions.set(selected+'::cuda',{x:0,y:0});['cuda_jl','ka','jacc'].forEach((m,i)=>positions.set(selected+'::'+m,{x:300,y:(i-1)*140}));return;
 }
 if($('scope').value==='neighborhood'){
  positions.set(selected,{x:0,y:0});const others=visibleNodes.filter(n=>n.id!==selected);
  others.forEach((n,i)=>{const angle=2*Math.PI*i/Math.max(1,others.length)-Math.PI/2;const r=210+20*(i%2);positions.set(n.id,{x:r*Math.cos(angle),y:r*Math.sin(angle)});});
 }else{
  families.forEach((f,i)=>{
   const members=visibleNodes.filter(n=>n.family===f).sort((a,b)=>a.id.localeCompare(b.id));const cx=(i%4)*370,cy=Math.floor(i/4)*350;
   members.forEach((n,j)=>{const theta=j*2.39996323,r=18*Math.sqrt(j);positions.set(n.id,{x:cx+Math.cos(theta)*r,y:cy+Math.sin(theta)*r});});
  });
 }
}
function resize(){const rect=canvas.getBoundingClientRect();width=rect.width;height=rect.height;const ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);fitGraph();}
function fitGraph(){
 const p=[...positions.values()];if(!p.length){scale=1;pan={x:width/2,y:height/2};draw();return;}
 const xs=p.map(v=>v.x),ys=p.map(v=>v.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
 scale=Math.min((width-150)/Math.max(150,maxX-minX),(height-100)/Math.max(150,maxY-minY),1.8);scale=Math.max(.1,scale);pan={x:width/2-(minX+maxX)/2*scale,y:height/2-(minY+maxY)/2*scale};draw();
}
function screen(id){const p=positions.get(id);return p?{x:p.x*scale+pan.x,y:p.y*scale+pan.y}:null;}
function nodeColor(n){if(n.model)return MODEL_COLORS[n.model];if(view()==='implementation')return CLASS_COLORS[n.models[model()].class]||'#7a8992';if(view()==='verification')return STATUS_COLORS[n.models[model()].status]||'#7a8992';return familyColor.get(n.family)||'#7a8992';}
function draw(){
 ctx.clearRect(0,0,width,height);
 visibleEdges.forEach(e=>{
  const a=screen(e.source),b=screen(e.target);if(!a||!b)return;
  const hot=selectedEdge===e||e.source===selected||e.target===selected;
  ctx.beginPath();ctx.strokeStyle=selectedEdge===e?'#152f49':view()==='translation'?MODEL_COLORS[e.model]:hot?'#809db6':'#d9e2e8';ctx.lineWidth=selectedEdge===e?2.5:hot?1.3:.8;
  ctx.setLineDash(e.status==='hardware_blocked'?[5,4]:[]);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
 });ctx.setLineDash([]);
 visibleNodes.forEach(n=>{
  const p=screen(n.id);if(!p)return;const hot=n.id===selected||n.baseId===selected;const radius=n.model?11:hot?9:visibleNodes.length>80?4.5:7;
  ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fillStyle=nodeColor(n);ctx.fill();
  const status=n.model&&n.model!=='cuda'?nodesById.get(n.baseId).models[n.model].status:n.models?.[model()].status;
  ctx.strokeStyle=hot?'#142e42':status==='hardware_blocked'?'#875e21':'#fff';ctx.lineWidth=hot?2:1;ctx.stroke();
  if($('labels').checked&&(visibleNodes.length<=45||n.id===selected||n.model)){
   const label=displayLabel(n);ctx.font=(hot?'600 ':'')+'12px system-ui';ctx.textAlign='center';ctx.fillStyle='#253742';ctx.fillText(label,p.x,p.y+radius+17);
  }
 });
 if(!visibleNodes.length){ctx.fillStyle='#65727c';ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillText('No nodes match these filters',width/2,height/2);}
}
function link(path,label){return path?`<a href="../../${encodeURI(path)}" target="_blank" rel="noopener">${esc(label)}</a>`:'';}
function detail(fields){return '<dl class="detail">'+fields.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(text(v))}</dd>`).join('')+'</dl>';}
function renderSelection(){
 const n=nodesById.get(selected),m=n.models[model()];
 $('selection').innerHTML=`<h3 class="selected-name">${esc(n.id)}</h3><span class="pill">${esc(n.family.replaceAll('_',' '))}</span><p>${esc(n.task||'Scientific task unknown')}</p>`+detail([['Method',n.method],['Accuracy',n.accuracy],['Named scale',n.size]])+`<p class="cap">${link(n.domain_path,'Scientific intent record')} · <a href="ontology_report.html">All property categories</a></p><div class="table-scroll"><table><thead><tr><th>Julia version</th><th>Scoped outcome</th><th>Implementation class</th></tr></thead><tbody>${['cuda_jl','ka','jacc'].map(k=>`<tr><td><button type="button" class="node-button model-name ${k}" data-inspect-model="${k}">${LABELS[k]}</button></td><td>${esc(STATUSES[n.models[k].status]||n.models[k].status)}</td><td>${esc(text(n.models[k].class))}</td></tr>`).join('')}</tbody></table></div><details class="summary-model"><summary>${LABELS[model()]} properties and evidence</summary>`+detail([['V100 profile status',m.roofline?.profiling_status],['V100 evidence scope',m.roofline?.positioning_evidence_scope],['Sampled kernel regimes',m.roofline?.geometric_regimes],['V100 causal bottleneck',m.roofline?.causal_bottleneck],['Algorithm',m.algorithm],['Storage',m.storage],['Compute',m.compute],['Mapping',m.mapping],['Layout',m.layout],['Resources',m.resources],['Device APIs',m.api],['Library',m.library],['Host scope',m.host],['Timing scope',m.timing_scope],['Profiling',m.profiling],['Actual failures',m.failure_codes],['History',m.failure_history],['Attempts',m.effort.candidate_attempts],['Compiler checks',m.effort.compiler_checks],['Token accounting',m.token_status],['Recorded total tokens',m.token_total],['Source-token proxy',m.saved_source_token_proxy],['Versions (defaults)',m.environment],['Review',m.review],['Class basis',m.class_basis]])+`<p>${link(m.record_path,'Implementation record')}</p></details><details class="summary-model"><summary>Exact configurations (${m.configurations.length})</summary><p class="cap">Includes trial verdicts, mean seconds from trials 2–5, reference qualifications and declared environment overrides. These are recorded values.</p><pre>${esc(JSON.stringify(m.configurations,null,2))}</pre></details><details class="summary-model"><summary>Origin-matched failure evidence</summary><pre>${esc(JSON.stringify(m.failure_events,null,2))}</pre></details><details class="summary-model"><summary>Observed token windows</summary><p class="cap">Windows can overlap and include repeated/cached context. Do not add overlapping windows or interpret source proxies as inference consumption.</p><pre>${esc(JSON.stringify(m.token_windows,null,2))}</pre></details><details class="summary-model"><summary>V100 evidence: native CUDA and selected Julia version</summary><p class="cap">Per-kernel geometric positions can be partial or warmup samples. They do not establish a whole-workflow bottleneck. Measurement failures are separate from translation failures.</p><pre>${esc(JSON.stringify({native_cuda:n.native_roofline,selected_Julia:m.roofline},null,2))}</pre></details>`;
}
function clearEvidence(){$('evidence').innerHTML='<p class="muted">Select a connection to inspect its shared properties, contributions and sources.</p>';}
function inspectEdge(e){
 selectedEdge=e;
 const baseSource=e.source.split('::')[0],baseTarget=e.target.split('::')[0],a=nodesById.get(baseSource),b=nodesById.get(baseTarget);
 const evidenceLinks=e.view==='science'?[link(a.domain_path,a.id+' scientific record'),link(b.domain_path,b.id+' scientific record')]:e.view==='translation'?[link(a.domain_path,'Native/scientific reference provenance'),link(b.models[e.model].record_path,LABELS[e.model]+' implementation record')]:[link(a.models[e.model].record_path,a.id+' '+LABELS[e.model]),link(b.models[e.model].record_path,b.id+' '+LABELS[e.model])];
 if(e.view==='implementation'&&DATA.meta.class_source)evidenceLinks.push(link(DATA.meta.class_source,'Reviewed class reconciliation'));
 $('evidence').innerHTML=`<strong>${esc(e.source)} ↔ ${esc(e.target)}</strong><p class="cap">${e.model==='shared'?'Reviewed scientific intent':LABELS[e.model]} · ${e.score===null?'Reference/implementation link':'Descriptive score '+e.score.toFixed(3)}</p>`+e.evidence.map(p=>`<div class="evidence-part"><strong>${esc(p.property)}</strong><div>${esc(text(p.shared))}</div>${p.contribution!==undefined?`<div class="muted">Score contribution ${p.contribution.toFixed(4)}; maximum weight ${p.weight.toFixed(2)}</div>`:''}${p.comparison?`<div class="muted">${esc(p.comparison)}</div>`:''}</div>`).join('')+`<p>${evidenceLinks.join(' · ')}</p><p class="cap">Shared descriptors support exploration. They do not establish semantic equivalence, a causal mechanism, equal timing or cross-vendor portability.</p>`;
 draw();
}
function renderEdges(){
 $('relationshipsTitle').textContent='Visible relationships ('+visibleEdges.length+')';
 const sorted=[...visibleEdges].sort((a,b)=>(b.score??0)-(a.score??0)||(a.source+a.target).localeCompare(b.source+b.target));
 $('edgeList').innerHTML=sorted.map(e=>`<button type="button" class="edge-button" data-edge="${DATA.edges.indexOf(e)}" data-source="${esc(e.source)}" data-target="${esc(e.target)}"><strong>${esc(e.source)} ↔ ${esc(e.target)}</strong><span>${e.score===null?'Recorded reference/implementation link':(e.model==='shared'?'Scientific':LABELS[e.model])+' · '+e.score.toFixed(3)}</span></button>`).join('')||( '<p class="empty">No retained connections meet these filters. Benchmarks without a connection are still assessed records.</p>');
}
function renderLegend(){
 let values;
 if(view()==='translation')values=Object.entries(MODEL_COLORS).map(([k,color])=>({key:k,label:LABELS[k],color}));
 else if(view()==='science')values=[...new Set(visibleNodes.map(n=>n.family))].sort().map(k=>({key:k,label:k.replaceAll('_',' '),color:familyColor.get(k)}));
 else if(view()==='implementation')values=[...new Set(visibleNodes.map(n=>n.models[model()].class||'unknown'))].sort().map(k=>({key:k,label:k.replaceAll('_',' '),color:CLASS_COLORS[k]||'#7a8992'}));
 else values=[...new Set(visibleNodes.map(n=>n.models[model()].status))].sort().map(k=>({key:k,label:STATUSES[k]||k,color:STATUS_COLORS[k]}));
 $('legend').innerHTML=values.map(v=>`<span><i style="background:${v.color}"></i>${esc(v.label)}</span>`).join('');
}
function hitNode(p){let closest=null,best=15;visibleNodes.forEach(n=>{const s=screen(n.id);if(!s)return;const d=Math.hypot(p.x-s.x,p.y-s.y);if(d<best){best=d;closest=n;}});return closest;}
function hitEdge(p){let best=6,edge=null;visibleEdges.forEach(e=>{const a=screen(e.source),b=screen(e.target);if(!a||!b)return;const dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy||1)));const d=Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy);if(d<best){best=d;edge=e;}});return edge;}
function local(ev){const r=canvas.getBoundingClientRect();return{x:ev.clientX-r.left,y:ev.clientY-r.top};}
canvas.onpointerdown=ev=>{canvas.setPointerCapture(ev.pointerId);pointer={...local(ev),start:local(ev)};moved=false;};
canvas.onpointermove=ev=>{const p=local(ev);if(pointer){const dx=p.x-pointer.x,dy=p.y-pointer.y;pan.x+=dx;pan.y+=dy;if(Math.hypot(p.x-pointer.start.x,p.y-pointer.start.y)>4)moved=true;pointer.x=p.x;pointer.y=p.y;draw();}else{const n=hitNode(p);canvas.title=n?displayLabel(n):'';canvas.style.cursor=n?'pointer':'grab';}};
canvas.onpointerup=ev=>{if(!moved){const p=local(ev),n=hitNode(p);if(n){selected=n.baseId||n.id;if(n.model&&n.model!=='cuda')$('model').value=n.model;$('search').value='';benchmarkChoices();populateFilters();showGraph();}else{const e=hitEdge(p);if(e)inspectEdge(e);}}pointer=null;};
canvas.onpointercancel=()=>{pointer=null;};
canvas.addEventListener('wheel',ev=>{ev.preventDefault();const p=local(ev),x=(p.x-pan.x)/scale,y=(p.y-pan.y)/scale;scale=Math.max(.08,Math.min(8,scale*Math.exp(-ev.deltaY*.001)));pan={x:p.x-x*scale,y:p.y-y*scale};draw();},{passive:false});
$('edgeList').onclick=ev=>{const b=ev.target.closest('[data-source]');if(!b)return;const e=visibleEdges.find(e=>e.source===b.dataset.source&&e.target===b.dataset.target);if(e)inspectEdge(e);};
$('selection').onclick=ev=>{const b=ev.target.closest('[data-inspect-model]');if(!b)return;$('model').value=b.dataset.inspectModel;populateFilters();showGraph();};
$('view').onchange=showGraph;$('model').onchange=()=>{populateFilters();benchmarkChoices();showGraph();};
$('search').oninput=()=>{benchmarkChoices();showGraph();};$('benchmark').onchange=()=>{if($('benchmark').value)selected=$('benchmark').value;showGraph();};
for(const id of ['scope','family','status','implClass','failure','profileStatus','profileRegime'])$(id).onchange=showGraph;
for(const id of ['score','neighbors','precision'])$(id).oninput=showGraph;
$('labels').onchange=draw;$('fit').onclick=fitGraph;
$('clear').onclick=()=>{$('search').value='';for(const id of ['family','status','implClass','precision','failure','profileStatus','profileRegime'])$(id).value='';$('score').value='0';$('neighbors').value='12';benchmarkChoices();showGraph();};
$('export').onclick=()=>{
 const documentData={generated_at:new Date().toISOString(),study:DATA.meta,selection:{benchmark:selected,model:model(),view:view(),scope:view()==='translation'?'selected_benchmark':$('scope').value,min_score:Number($('score').value)},nodes:visibleNodes.map(n=>n.model?{...n,record:n.model==='cuda'?nodesById.get(n.baseId).native_roofline:nodesById.get(n.baseId).models[n.model]}:n),edges:visibleEdges};
 const url=URL.createObjectURL(new Blob([JSON.stringify(documentData,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='hecbench-julia-relationships.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('stats').innerHTML=[['Assessed benchmarks',DATA.meta.assessed_names,'All reviewed outcomes retained'],['Julia implementation records',DATA.meta.implementation_records,'CUDA.jl · KA · JACC'],['Scientific families',families.length,'Source-reviewed intent'],['Retained descriptive connections',DATA.edges.length,`Sparse union of top-${DATA.meta.top_k_per_endpoint} neighbors`]].map(([k,v,s])=>`<div class="stat">${esc(k)}<strong>${v.toLocaleString()}</strong><span>${esc(s)}</span></div>`).join('');
$('provenance').textContent=`Input fingerprint: ${DATA.meta.source_sha256}. Class vocabulary source: ${DATA.meta.class_source||'unavailable; raw declarations retained'}.`;
$('footer').textContent=`Generated ${DATA.meta.generated_at}. Source records: ${DATA.meta.study_path}. Derived graph: ontology/relationships/julia_relationships.json. Original catalogue similarity edges are preserved.`;
populateFilters();benchmarkChoices();showGraph();resize();new ResizeObserver(resize).observe(canvas);
