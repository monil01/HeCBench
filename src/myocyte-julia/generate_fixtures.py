
from pathlib import Path
import struct,json,hashlib
p=Path(__file__).resolve().parent
y=list(map(float,(p/"study_y.txt").read_text().split()));params=list(map(float,(p/"study_params.txt").read_text().split()));params+=params[-1:]*(18-len(params))
values=[]
for case in range(4):
 state=y.copy();par=params.copy();time=[0.,6.75,5.75,1006.25][case]
 if case==1:
  state[38]=-35.;state[37]*=1.05;par[15]=1000.;par[16]=135.;par[17]=1.5
 elif case==2:
  state[38]=-82.;state[46:]=[v*.97 for v in state[46:]];par[15]=1000.;par[16]=100.;par[17]=.5
 elif case==3:
  for i in range(3):state[i]+=.001
  state[35]*=1.1;state[36]*=.95;state[37]*=1.05;par[15]=1000.;par[16]=135.;par[17]=1.2
 values+=state+par+[time,float(int(time))]
data=struct.pack("<"+"f"*len(values),*values);(p/"study_probes_f32.bin").write_bytes(data)
provenance={"count":len(values),"dtype":"float32","byte_order":"little","cases":4,"percase_layout":"state91/params18/stagetime/truncatedkernelinttime","source_parameter_EOF_repeat_last":True,"source_y_sha256":hashlib.sha256((p/"study_y.txt").read_bytes()).hexdigest(),"source_params_sha256":hashlib.sha256((p/"study_params.txt").read_bytes()).hexdigest(),"fixtures":{"study_probes_f32.bin":{"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),"CPU_generated":True}},"scope":"case0sourceinitial+EOFparams; threeexplicitphysicalstate/parameter/timeperturbations cover voltagebranch/Mgbranch/pacingonoff; notcalibratedphysiologicalpatientdata"}
(p/"fixture_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
