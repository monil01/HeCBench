from pathlib import Path
import difflib,hashlib,json
R=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent;P=Path('/tmp/roofline-completion-20261021/profiler_method_presentation_v2/proposed/tools/v100_roofline_refresh.py');sha=lambda b:hashlib.sha256(b).hexdigest();base=(R/'tools/v100_roofline_refresh.py').read_text();future=P.read_text()
def refine(s):
 a="    ap.add_argument('--once',action='store_true')";b=a+"\n    ap.add_argument('--current537-display-review-sha256',help='Pass exact root-approved current537 display authority SHA to report generation only.')";assert a in s;s=s.replace(a,b,1)
 a="            if run('ontology_report',[sys.executable,str(root/'ontology/tools/generate_ontology_report.py'),'--population','537','--output',str(temporary)]):"
 b="            report_command=[sys.executable,str(root/'ontology/tools/generate_ontology_report.py'),'--population','537','--output',str(temporary)]\n            if args.current537_display_review_sha256:report_command+=['--current537-display-review-sha256',args.current537_display_review_sha256]\n            if run('ontology_report',report_command):";assert a in s;s=s.replace(a,b,1);return s
small=refine(base);composed=refine(future)
for group,s in [('base',base),('proposed_flag_only',small),('future_method_base',future),('composed_flag_only',composed)]:
 p=O/group/'tools/v100_roofline_refresh.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
for name,a,b in [('minimal_passthrough.patch',base,small),('composed_vs_live.patch',base,composed),('composed_vs_method_v2.patch',future,composed)]:
 (O/name).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/tools/v100_roofline_refresh.py',tofile='b/tools/v100_roofline_refresh.py')))
(O/'bindings.json').write_text(json.dumps({'live_base_sha256':sha(base.encode()),'flag_only_sha256':sha(small.encode()),'future_method_v2_path':str(P),'future_method_v2_sha256':sha(future.encode()),'composed_flag_only_sha256':sha(composed.encode()),'execution_status':'NOT_EXECUTED_NOT_APPLIED','known_hold':'Flag-only candidates deadlock with current private loader LOCK_SH while refresh waits under LOCK_EX. Companion inherited-FD lock protocol must be reviewed/adopted first.'},indent=2)+'\n')
