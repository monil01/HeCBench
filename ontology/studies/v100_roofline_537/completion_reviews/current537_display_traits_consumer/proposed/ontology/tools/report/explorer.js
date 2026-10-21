/* Scoped explorer: catalogue properties are shared; study properties belong to one model row. */
const MODEL_LABELS={cuda_jl:'CUDA.jl',ka:'KernelAbstractions',jacc:'JACC',catalogue:'Catalogue only'};
const COLS=STUDY.columns;
const GROUPS=[...new Set(COLS.map(c=>c.group))];
const EXPLORER=STUDY.rows;
const BY_ID=new Map(DATA.map(r=>[r.id,r]));
const CATALOGUE_FACETS=new Set(['type','domain','suite','algo','motif','pattern','bias','precision','flags','kbin','backends','nbs','validation','code','mem','depkind']);
const FL={type:'Type',domain:'Catalogue domain',suite:'Suite',algo:'Algorithm',motif:'Motif',pattern:'Pattern',bias:'Inferred resource bias',precision:'Precision',flags:'Flag',kbin:'Kernel count',backends:'Backend',nbs:'Backends',validation:'Catalogue validation',code:'Code structure',mem:'Memory model',depkind:'Dependency'};
let facets={},filters=[],visible=new Set(COLS.map(c=>c.key)),sortK='id',sortDir=1,page=0,filtered=[];
const columnsByKey=new Map(COLS.map(c=>[c.key,c]));
const rawKeys=new Set();
Object.values(STUDY.base_properties).forEach(p=>Object.keys(p).forEach(k=>rawKeys.add(k)));
EXPLORER.forEach(r=>Object.keys(r.properties).forEach(k=>rawKeys.add(k)));
const propertyOptions=[...COLS.map(c=>({key:'column:'+c.key,label:c.group+' · '+c.label})),...[...rawKeys].sort().map(k=>({key:'raw:'+k,label:k}))];
const propertiesByKey=new Map(propertyOptions.map(p=>[p.key,p]));
function scalars(value){
  if(value===null||value===undefined)return [];
  if(Array.isArray(value))return value.flatMap(scalars);
  if(typeof value==='object')return Object.values(value).flatMap(scalars);
  return [value];
}
function isKnown(v){return v!==null&&v!==undefined&&!['','unknown','not_recorded','not recorded','uncomputed'].includes(String(v).trim().toLowerCase());}
function propValues(row,key){
  if(key.startsWith('column:'))return scalars(row.values[key.slice(7)]);
  const path=key.slice(4);
  return row.properties[path] ?? STUDY.base_properties[row.id]?.[path] ?? [];
}
function matches(row,f){
  const values=propValues(row,f.key);const known=values.filter(isKnown);
  const needle=String(f.value??'').toLocaleLowerCase();
  if(f.op==='known')return known.length>0;
  if(f.op==='unknown')return known.length===0;
  if(f.op==='contains')return values.some(v=>String(v).toLocaleLowerCase().includes(needle));
  if(f.op==='eq')return values.some(v=>String(v).toLocaleLowerCase()===needle);
  if(f.op==='neq')return known.length>0&&!values.some(v=>String(v).toLocaleLowerCase()===needle);
  const n=Number(f.value);
  // Boolean values, empty strings and missing metrics are never numeric zero.
  return known.some(v=>typeof v!=='boolean'&&String(v).trim()!==''&&Number.isFinite(Number(v))&&(f.op==='gte'?Number(v)>=n:Number(v)<=n));
}
function columnValue(row,col){return col.rawKey?propValues(row,col.rawKey):row.values[col.key];}
function showValue(v){
  if(v===null||v===undefined||(Array.isArray(v)&&!v.length))return 'Unknown';
  if(typeof v==='object'&&!Array.isArray(v))return JSON.stringify(v);
  if(Array.isArray(v))return v.map(x=>typeof x==='object'&&x!==null?JSON.stringify(x):String(x)).join(' · ');
  return String(v);
}
function setFacet(key,val,_label,silent){
  facets[key]=val;page=0;
  if(!silent){render();$('explorer').scrollIntoView({behavior:'smooth',block:'start'});}
}
function buildColumnControls(){
  $('columnGroups').innerHTML='<div class="column-actions"><button type="button" id="allColumns">Show all</button><button type="button" id="essentialColumns">Compact view</button></div>'+GROUPS.map((g,i)=>{
    const cols=COLS.filter(c=>c.group===g);
    return `<details class="column-group"><summary>${esc(g)} (${cols.length})</summary><label><input type="checkbox" data-group="${i}" checked> All in category</label>${cols.map(c=>`<label><input type="checkbox" data-column="${c.key}" checked ${c.key==='id'?'disabled':''}>${esc(c.label)}</label>`).join('')}</details>`;
  }).join('');
  $('columnGroups').onchange=e=>{
    const target=e.target;
    if(target.dataset.group!==undefined){const group=GROUPS[Number(target.dataset.group)];COLS.filter(c=>c.group===group&&c.key!=='id').forEach(c=>target.checked?visible.add(c.key):visible.delete(c.key));}
    else if(target.dataset.column){target.checked?visible.add(target.dataset.column):visible.delete(target.dataset.column);}
    render();
  };
  $('allColumns').onclick=()=>{visible=new Set(COLS.map(c=>c.key));render();};
  $('essentialColumns').onclick=()=>{visible=new Set(['id','model','status','family','algorithm','storage','layout','mapping','resources','class','agreement','times','attempts','julia','related']);render();};
}
function refreshColumnControls(){
  document.querySelectorAll('[data-column]').forEach(e=>e.checked=visible.has(e.dataset.column));
  document.querySelectorAll('[data-group]').forEach(e=>{
    const keys=COLS.filter(c=>c.group===GROUPS[Number(e.dataset.group)]).map(c=>c.key);const n=keys.filter(k=>visible.has(k)).length;e.checked=n===keys.length;e.indeterminate=n>0&&n<keys.length;
  });
  $('columnCount').textContent=`(${visible.size} of ${COLS.length} shown; ${GROUPS.length} categories)`;
}
function updatePropertyPicker(){
  const needle=$('propertySearch').value.trim().toLowerCase();const current=$('propertySelect').value;
  const matched=propertyOptions.filter(p=>!needle?p.key.startsWith('column:'):p.label.toLowerCase().includes(needle));
  const shown=matched.slice(0,300);
  $('propertySelect').innerHTML=shown.map(p=>`<option value="${esc(p.key)}">${esc(p.label)}</option>`).join('');
  if(shown.some(p=>p.key===current))$('propertySelect').value=current;
  $('propertyHelp').textContent=`${propertyOptions.length.toLocaleString()} filterable fields. ${needle?matched.length+' match your property search.':'Search to access every raw record field, including nested configuration properties.'}${matched.length>300?' Showing the first 300 matches; narrow your search.':''} Numeric conditions match any recorded number. Multiple filters are combined with AND.`;
  updateValueSuggestions();
}
function updateValueSuggestions(){
  const key=$('propertySelect').value;
  const values=new Set(EXPLORER.flatMap(r=>propValues(r,key)).filter(isKnown).map(String));
  $('propertyValues').innerHTML=[...values].filter(v=>v.length<160).sort().slice(0,400).map(v=>`<option value="${esc(v)}"></option>`).join('');
}
function addFilter(){
  const key=$('propertySelect').value,op=$('filterOp').value,value=$('filterValue').value.trim();
  $('filterError').textContent='';
  if(!key){$('filterError').textContent='Choose a property first.';return;}
  if(!['known','unknown'].includes(op)&&!value){$('filterError').textContent='Enter a value for this condition.';return;}
  if(['gte','lte'].includes(op)&&!Number.isFinite(Number(value))){$('filterError').textContent='Enter a finite number.';return;}
  filters.push({key,op,value});page=0;render();
}
function clearFilters(){facets={};filters=[];page=0;$('q').value='';$('modelFilter').value='';$('filterError').textContent='';render();}
function sortable(v){const values=scalars(v).filter(isKnown);if(!values.length)return null;if(values.length===1&&typeof values[0]==='number')return values[0];return showValue(v).toLocaleLowerCase();}
function render(){
  const q=$('q').value.trim().toLocaleLowerCase();const model=$('modelFilter').value;
  filtered=EXPLORER.filter(r=>{
    if(model&&r.model!==model)return false;
    const base=BY_ID.get(r.id)||{};
    if(!Object.entries(facets).every(([k,v])=>[].concat(CATALOGUE_FACETS.has(k)?(base[k]??[]):(r.values[k]??[])).map(String).includes(String(v))))return false;
    if(!filters.every(f=>matches(r,f)))return false;
    if(q){
      if(!r.searchText)r.searchText=[r.id,MODEL_LABELS[r.model],JSON.stringify(r.values),JSON.stringify(r.properties),JSON.stringify(STUDY.base_properties[r.id])].join(' ').toLocaleLowerCase();
      if(!r.searchText.includes(q))return false;
    }
    return true;
  });
  filtered.sort((a,b)=>{const x=sortable(columnValue(a,columnsByKey.get(sortK))),y=sortable(columnValue(b,columnsByKey.get(sortK)));if(x===null)return y===null?0:1;if(y===null)return -1;return (typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y),undefined,{numeric:true}))*sortDir||a.model.localeCompare(b.model);});
  const size=Number($('pageSize').value);const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);
  const cols=COLS.filter(c=>visible.has(c.key));
  document.querySelector('#tbl thead').innerHTML='<tr>'+cols.map(c=>`<th aria-sort="${sortK===c.key?(sortDir===1?'ascending':'descending'):'none'}"><button type="button" data-sort="${c.key}">${esc(c.label)}${sortK===c.key?(sortDir===1?' ↑':' ↓'):''}</button></th>`).join('')+'</tr>';
  document.querySelector('#tbl tbody').innerHTML=filtered.slice(page*size,(page+1)*size).map(r=>'<tr>'+cols.map(c=>{
    let value=columnValue(r,c);if(c.key==='model')value=MODEL_LABELS[r.model];
    const text=showValue(value);const cell=c.key==='id'?`<button type="button" class="record-open" data-record="${esc(r.id)}" data-model="${r.model}">${esc(r.id)}</button>`:`<div class="cell-value${text==='Unknown'?' unknown':''}" title="${esc(text.slice(0,3000))}">${esc(text)}</div>`;
    return `<td>${cell}</td>`;
  }).join('')+'</tr>').join('')||( `<tr><td colspan="${cols.length}">No matching benchmarks. Remove a filter to broaden the results.</td></tr>`);
  const names=new Set(filtered.map(r=>r.id)).size;
  $('count').textContent=`${filtered.length.toLocaleString()} implementation rows · ${names} benchmark names (of ${EXPLORER.length.toLocaleString()} rows / ${N} catalogue names)`;
  $('pageLabel').textContent=`Page ${page+1} of ${pages}`;$('prevPage').disabled=page===0;$('nextPage').disabled=page>=pages-1;
  const labels={contains:'contains',eq:'equals',neq:'does not equal',gte:'≥',lte:'≤',known:'has recorded value',unknown:'is unknown / missing'};
  $('chips').innerHTML=Object.entries(facets).map(([k,v])=>`<span class="chip">${esc(FL[k]||k)}: ${esc(v)} <button type="button" data-facet="${esc(k)}" aria-label="Remove filter">×</button></span>`).join('')+filters.map((f,i)=>`<span class="chip">${esc(propertiesByKey.get(f.key)?.label||f.key)} ${labels[f.op]} ${['known','unknown'].includes(f.op)?'':esc(f.value)} <button type="button" data-filter="${i}" aria-label="Remove filter">×</button></span>`).join('');
  refreshColumnControls();
}
function openRecord(id,model){
  const row=EXPLORER.find(r=>r.id===id&&r.model===model);if(!row)return;
  $('recordTitle').textContent=`${id} — ${MODEL_LABELS[model]}`;
  $('recordLinks').innerHTML=`<a href="../../${encodeURI(row.record_path)}" target="_blank" rel="noopener">Implementation record</a> · <a href="../benchmarks/${encodeURIComponent(id)}.yaml" target="_blank" rel="noopener">Catalogue record</a>`+(model!=='catalogue'?` · <a href="../studies/julia_abstractions_447/domains/${encodeURIComponent(id)}.json" target="_blank" rel="noopener">Scientific record</a>`:'');
  const props={...(STUDY.base_properties[id]||{}),...row.properties};
  if(row.display_provenance?.provenance?.length){
    const display=row.display_provenance;
    $('recordLinks').innerHTML+=' · '+display.provenance.map(p=>`<a href="../../${encodeURI(p.kind==='curated_aliases'?p.domain_path:p.authority.overlay.path)}" target="_blank" rel="noopener">${esc(p.kind==='curated_aliases'?'Curated field source':'Current537 display overlay')}</a>`).join(' · ')+` · <a href="../../${encodeURI(display.provenance[0].authority.path)}" target="_blank" rel="noopener">Root display authority</a>`;
  }
  $('recordDetails').innerHTML=GROUPS.map(g=>`<details open><summary>${esc(g)}</summary><dl class="detail-list">${COLS.filter(c=>c.group===g).map(c=>`<dt>${esc(c.label)}</dt><dd>${esc(showValue(columnValue(row,c)))}</dd>`).join('')}</dl></details>`).join('')+`<details><summary>Exact recorded configurations (${row.configurations.length})</summary><pre>${esc(JSON.stringify(row.configurations,null,2))}</pre></details><details><summary>All recorded property paths (${Object.keys(props).length})</summary><dl class="detail-list">${Object.entries(props).sort(([a],[b])=>a.localeCompare(b)).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(showValue(v))}</dd>`).join('')}</dl></details>`;
  $('recordDialog').showModal();
}
function showPropertyColumn(){
  const key=$('propertySelect').value;if(!key)return;
  if(key.startsWith('column:'))visible.add(key.slice(7));
  else {
    const id='property:'+key.slice(4);let col=columnsByKey.get(id);
    if(!col){
      const group='Additional recorded properties';if(!GROUPS.includes(group))GROUPS.push(group);
      col={group,key:id,label:key.slice(4),rawKey:key};COLS.push(col);columnsByKey.set(id,col);
    }
    visible.add(id);buildColumnControls();
  }
  render();
}
function exportCsv(){
  const cols=COLS.filter(c=>visible.has(c.key));
  const csvCell=v=>{let s=showValue(v);if(/^[=+@-]/.test(s)&&typeof v!=='number')s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};
  const lines=[cols.map(c=>csvCell(c.label)).join(','),...filtered.map(r=>cols.map(c=>csvCell(c.key==='model'?MODEL_LABELS[r.model]:columnValue(r,c))).join(','))];
  const url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download='hecbench-julia-ontology.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function studyOverview(){
  const labels={verified_named_configuration:'Ordinary named contracts (includes property cases)',conditionally_verified_native_reference_repaired:'Corrected native reference',verified_native_behavior_known_scientific_defect:'Known native defect',conditionally_verified_revised_acceptance_contract:'Revised acceptance',execution_only_verified:'Execution only',hardware_blocked:'Native hardware barrier'};
  $('studyCards').innerHTML=[['Inventoried Julia names',STUDY.assessed_names,'Distinct catalogue names; outcomes stay qualified'],['Julia implementation records',STUDY.source_files,'CUDA.jl, KA and JACC'],['Explorer property columns',COLS.length,`${GROUPS.length} selectable categories`],['All filterable fields',propertyOptions.length,'Includes every recorded nested property path']].map(([k,v,d])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v.toLocaleString()}</div><div class="d">${d}</div></div>`).join('');
  const table=(keys,label,counts)=>'<table class="study-table"><thead><tr><th>'+label+'</th>'+Object.keys(MODEL_LABELS).filter(k=>k!=='catalogue').map(k=>`<th>${esc(MODEL_LABELS[k])}</th>`).join('')+'</tr></thead><tbody>'+keys.map(k=>`<tr><td>${esc(labels[k]||k.replaceAll('_',' '))}</td>${['cuda_jl','ka','jacc'].map(m=>`<td class="study-count">${counts[m]?.[k]??0}</td>`).join('')}</tr>`).join('')+'</tbody></table>';
  $('studyOutcomes').innerHTML=table(Object.keys(labels),'Scoped outcome',STUDY.status_counts);
  const classes={cuda_jl:{},ka:{},jacc:{}};
  EXPLORER.filter(r=>r.model!=='catalogue').forEach(r=>{const c=r.values.class||'unknown';classes[r.model][c]=(classes[r.model][c]||0)+1;});
  const families=new Map();EXPLORER.filter(r=>r.model==='cuda_jl').forEach(r=>{const f=r.values.family||'unknown';families.set(f,(families.get(f)||0)+1);});hbar('studyFamilies',[...families].sort((a,b)=>b[1]-a[1]),{key:'family',denom:STUDY.assessed_names,facetLabel:'Scientific family'});
  const keys=[...new Set(Object.values(classes).flatMap(c=>Object.keys(c)))].sort();$('studyClasses').innerHTML=table(keys,'Implementation class',classes);
}
buildColumnControls();updatePropertyPicker();studyOverview();
$('q').oninput=()=>{page=0;render();};$('modelFilter').onchange=()=>{page=0;render();};
$('propertySearch').oninput=updatePropertyPicker;$('propertySelect').onchange=updateValueSuggestions;$('addFilter').onclick=addFilter;
$('filterOp').onchange=()=>{$('filterValue').disabled=['known','unknown'].includes($('filterOp').value);};
$('showPropertyColumn').onclick=showPropertyColumn;
$('filterValue').onkeydown=e=>{if(e.key==='Enter')addFilter();};$('resetFilters').onclick=clearFilters;$('exportCsv').onclick=exportCsv;
$('pageSize').onchange=()=>{page=0;render();};$('prevPage').onclick=()=>{page--;render();};$('nextPage').onclick=()=>{page++;render();};
$('chips').onclick=e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.facet!==undefined)delete facets[b.dataset.facet];if(b.dataset.filter!==undefined)filters.splice(Number(b.dataset.filter),1);page=0;render();};
$('tbl').onclick=e=>{const s=e.target.closest('[data-sort]'),r=e.target.closest('[data-record]');if(s){sortDir=sortK===s.dataset.sort?-sortDir:1;sortK=s.dataset.sort;render();}else if(r)openRecord(r.dataset.record,r.dataset.model);};
$('closeDialog').onclick=()=>$('recordDialog').close();
render();
$('foot').textContent=`Generated ${META.generated}. ${N} catalogue YAML records; ${STUDY.assessed_names} scientific records and ${STUDY.source_files} Julia implementation records in ${STUDY.study_path}. Self-contained: search, filters and charts work offline. Token windows can overlap; source-token proxies do not measure inference consumption. Trial times and means retain their configuration-specific scope. Source fingerprint: ${STUDY.source_digest.slice(0,16)}.`;
