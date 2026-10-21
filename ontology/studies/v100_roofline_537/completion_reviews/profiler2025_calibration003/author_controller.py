"""Private text-only003 authoring; never import/execute controller."""
from pathlib import Path
import json,hashlib,difflib
O=Path(__file__).parent;PRE=O.parent/'profiler2025_calibration002';h=lambda b:hashlib.sha256(b).hexdigest()
old=(PRE/'calibrate_controller.py').read_bytes()
if h(old)!='7211c35d2d5fc02e157b1142622cc31c6fd78510685e3b8244f15f5d675caf26':raise ValueError('002 source drift')
s=old.decode()
a="  for m in metrics:\n   u=units.get(m)"
b="  unit_aliases=[]\n  frequency_metrics={'dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__cycles_elapsed.avg.per_second'}\n  for m in metrics:\n   u=units.get(m)"
if s.count(a)!=1:raise ValueError('Ambiguous source')
s=s.replace(a,b,1)
a="   elif expectedunit and u!=expectedunit:raise RuntimeError('Metric unit mismatch '+m+' '+str(u)+' expected '+expectedunit)"
b="""   elif m in frequency_metrics:
    if u not in ('cycle/second','hz'):raise RuntimeError('Exact frequency unit mismatch '+m+' '+str(u))
    if u=='hz':unit_aliases.append({'metric':m,'raw_unit':'hz','validated_unit':'cycle/second','factor':1,'scope':'Only this exact elapsed-cycle frequency metric, --print-units base. Raw counters unchanged.'})
   elif expectedunit and u!=expectedunit:raise RuntimeError('Metric unit mismatch '+m+' '+str(u)+' expected '+expectedunit)"""
if s.count(a)!=1:raise ValueError('Ambiguous unit edit')
s=s.replace(a,b,1)
a="'actual_PCI':pci,'all_requested_metrics_present':True}"
b="'actual_PCI':pci,'all_requested_metrics_present':True,'validated_frequency_unit_aliases':unit_aliases}"
if s.count(a)!=1:raise ValueError('Ambiguous evidence edit')
s=s.replace(a,b,1);(O/'calibrate_controller.py').write_text(s)
(O/'002_to_003_controller.patch').write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),s.splitlines(True),fromfile='002/calibrate_controller.py',tofile='003/calibrate_controller.py')))
# Equivalent authoring script with fresh own paths and003 output only.
p=(PRE/'prepare_manifest.py').read_text();p=p.replace("method_calibration_ncu2025_2_0_002","method_calibration_ncu2025_2_0_003")
(O/'prepare_manifest.py').write_text(p)
print(json.dumps({'source':str(O/'calibrate_controller.py'),'source_sha256':h(s.encode()),'no_controller_execution':True}))
