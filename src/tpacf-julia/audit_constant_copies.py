#!/usr/bin/env python3
from pathlib import Path
import argparse,json,re,hashlib
p=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--result',required=True);ap.add_argument('--receipt',required=True);a=ap.parse_args();rp=Path(a.result);s=(rp.parent/(a.model+'_run1.log')).read_text();expected=(p/'data/source_boundaries63.bin').read_bytes()*2;rows=[]
for line in s.splitlines():
 if not line.startswith('TPACF_CONSTANT_COPIES '):continue
 d=dict(re.findall(r'(\w+)=([^ ]+)',line));f=Path(d['path']);b=f.read_bytes();assert int(d['modules'])==2 and b==expected and hashlib.sha256(b).hexdigest()==d['sha256'];rows.append({'trial':int(d['trial']),'case':int(d['case']),'path':str(f),'sha256':d['sha256'],'bytes':len(b)})
assert len(rows)==18;assert sorted((r['trial'],r['case'])for r in rows)==[(t,c)for t in range(6)for c in range(3)]
r={'model':a.model,'result':str(rp),'result_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'actual_module_constant_copies_per_case':2,'full_actual_constant_bytes_checked_all5':15120,'raw_complete_files_including_warmup':rows,'source_constant_symbol':'tpacf_study_bins','PTX_actual_AS4_declaration_present':'.const' in s and 'tpacf_study_bins' in s,'explicit_PTX_reflection_requests_this_process':s.count('TPACF_PTX_BEGIN_SYM'),'compiler_API_note':'Factoryalso2launchfalseACFcompilation/cachelookups/case; notsourceiterations, actualbackendinvocationsunknown','issues':[]};Path(a.receipt).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='raw_complete_files_including_warmup'}))
