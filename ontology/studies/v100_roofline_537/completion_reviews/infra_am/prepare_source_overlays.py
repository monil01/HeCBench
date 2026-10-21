#!/usr/bin/env python3
"""Prepare reviewed host/output-only copies; never compile or run benchmarks.

This generates a source packet, NOT an authorized executable profile plan. Parent
must bind packet/diffs and canonical inputs into a fresh guarded collector plan.
"""
import argparse,difflib,hashlib,json,pathlib,re
S=pathlib.Path('/home/imo/HeCBench_julia')
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--case',required=True);a.add_argument('--destination',type=pathlib.Path,required=True);z=a.parse_args();spec=json.loads((D/'source_overlay_proposals.json').read_text());case=next(x for x in spec['cases']if x['case']==z.case);out=z.destination.resolve()
 if not out.is_relative_to('/tmp')or out==pathlib.Path('/tmp')or out.exists():raise SystemExit('Require nonexistent fresh /tmp destination')
 sources=[]
 for f in case['files']:
  p=pathlib.Path(f['path'])
  if sha(p)!=f['sha256']:raise SystemExit('Source hash changed: '+str(p))
  text=p.read_text();original=text
  for r in f['replacements']:
   if text.count(r['old'])!=r.get('count',1):raise SystemExit('Replacement cardinality mismatch: '+r['old'])
   new=r['new'].replace('${OUTPUT}',str(out/'profile_outputs')).replace('${SCRATCH}',str(out))
   text=text.replace(r['old'],new)
  # Copy-only C/C++ include lookup is pinned to exact original canonical files.
  includes=[]
  if p.suffix in ['.cu','.cpp','.cuh','.h']:
   def canonical(m):
    target=(p.parent/m.group(2)).resolve()
    if not target.is_file():return m.group(0)
    includes.append({'path':str(target),'sha256':sha(target)})
    return m.group(1)+'"'+str(target)+'"'
   text=re.sub(r'(^\s*#\s*include\s*)"([^"]+)"',canonical,text,flags=re.M)
  if p.suffix=='.jl':
   def canonicalj(m):
    target=(p.parent/m.group(1)).resolve()
    if not target.is_file():raise ValueError('Relative include missing: '+str(target))
    includes.append({'path':str(target),'sha256':sha(target)});return 'include('+json.dumps(str(target))+')'
   text=re.sub(r'include\("([^"\n]+)"\)',canonicalj,text)
  sources.append((p,text,original,includes))
 out.mkdir(parents=True);(out/'profile_outputs').mkdir()
 receipts=[]
 for i,(p,text,original,includes)in enumerate(sources):
  q=out/(str(i)+'_'+p.name);q.write_text(text);diff=out/(q.name+'.diff');diff.write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(p),tofile=str(q))))
  receipts.append({'source':str(p),'source_sha256_before':sha(p),'copy':str(q),'copy_sha256':sha(q),'diff':str(diff),'diff_sha256':sha(diff),'canonical_read_only_includes':includes,'source_sha256_after':sha(p)})
 receipt={'case':z.case,'status':'source_packet_only_not_executable_plan','method':'Only host output paths and canonical read-only source lookup; no kernel body, data, arithmetic, dimensions, tolerance, iteration or timings modified.','original_selected_parents':case['parents'],'source_copies':receipts,'additional_precreated_dirs':[],'remaining_review':case.get('remaining_review',[]),'helper':str(pathlib.Path(__file__).resolve()),'helper_sha256':sha(pathlib.Path(__file__))}
 if z.case=='mf_sgd_native_precreate':raise RuntimeError('Not a source overlay: parent must precreate plan-owned argv[4]')
 (out/'source_overlay.json').write_text(json.dumps(receipt,indent=2)+'\n');print(out/'source_overlay.json')
if __name__=='__main__':main()
