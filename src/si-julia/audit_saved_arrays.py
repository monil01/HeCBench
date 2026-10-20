#!/usr/bin/env python3
"""Full actual inputs and every tile output against independent set intersections."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.result.read_text());c=json.loads(a.result.with_name('config.json').read_text());P=a.source_root/'src/si-julia';o=c['input_contract']['oracle'];expected=np.fromfile(P/o['path'],dtype='<f4');assert hashlib.sha256((P/o['path']).read_bytes()).hexdigest()==o['sha256'];assert len(expected)==o['total_count'];rows=[];groups={}
for model,m in r['models'].items():
 for t in m.get('trials',[]):
  art=t['output_artifact'];f=Path(art.get('saved_path',art['path']));h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==art['sha256'];actual=np.fromfile(f,dtype='<f4');assert actual.size==expected.size and np.isfinite(actual).all();assert np.array_equal(actual,expected),(model,t['trial'],np.flatnonzero(actual!=expected)[:12].tolist());groups.setdefault(t['trial'],[]).append(h);rows.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':h,'bytes':f.stat().st_size,'all_actual_device_membership_input_words_exact':True,'all_pair_intersection_counts_and_padding_guards_exact':True,'max_abs_error':float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),'scope':'EveryactualA/BmembershipmatrixcapturebeforeGEMMuse, allP2outputwords+17guards peruppertilepair; unusedpartialpaddingzeroonlyexplicitcorrectedsecondary'})
assert rows
for hashes in groups.values():assert len(set(hashes))==1
out={'status':'pass','result':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(a.result.with_name('config.json').read_bytes()).hexdigest(),'NumPy':np.__version__,'oracle':o,'oracle_kind':'CPUsetintersection, notmatrixmultiplication andnotnativealgorithmtranscription; predeclaredexactintegeroutputs/inputs/guards','artifact_count':len(rows),'artifact_total_bytes':sum(x['bytes']for x in rows),'raw_artifact_bytes_identical_all_models_per_trial':True,'rows':rows};a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k not in('rows','oracle')})
