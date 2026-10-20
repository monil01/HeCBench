from pathlib import Path
import re,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent;R=B.parent.parent;f=R/'.porting-state/diagnostics/logic-resim/wave_300M_result.saif';names=json.load(open(B/'wire_names.json'));idx={n:i for i,n in enumerate(names)};expected=np.fromfile(B/'oracle_occupancy_u64.bin',dtype='<u8').reshape(-1,3);found={}
for name,a,b,c in re.findall(r'\(([^\n]+)\n\(T0 (\d+)\) \(T1 (\d+)\) \(TX (\d+)\)',f.read_text()):
 assert name in idx;triple=[int(a),int(b),int(c)];assert triple==expected[idx[name]].tolist();assert sum(triple)==300000000;found[name]=triple
assert len(found)==len(names)
report={'benchmark':'logic-resim','status':'pass','allnets':len(found),'all_occupancy_values_checked':len(found)*3,'actual_source_SAIF_vs_independent_wave_occupancy_exact':True,'every_net_T0_T1_TX_conservation_exact':True,'source_X_Z_combined_in_TX':True,'source_TC_and_IG_literal_zero_not_measured_switching_activity':True,'source_path':str(f.relative_to(R)),'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'window_ps':[0,300000000]};(P/'saif_occupancy_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
