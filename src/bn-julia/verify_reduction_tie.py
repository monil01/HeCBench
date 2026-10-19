"""CPU counterexample for the original block winner trace on valid tied scores."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
def replay(left_tie):
 a=[-6.3984375]*163+[-999999999999999.0]*93
 for step in [128,64,32,16,8,4,2,1]:
  for tid in range(step):
   left,right=a[tid],a[tid+step]
   if right>left and right<0:a[tid]=right;a[tid+step]=float(tid+step)
   elif (right<=left if left_tie else right<left)and left<0:a[tid+step]=float(tid)
   elif left>0 and right<0:a[tid]=right;a[tid+step]=float(tid+step)
   elif left<0 and right>0:a[tid+step]=float(tid)
 t=0
 for i in range(7):
  if not 0<=t<128:break
  t=int(a[2**i+t])
 return {'winning_score':a[0],'trace_winner':t,'valid_thread':0<=t<256}
a=replay(False);b=replay(True);assert not a['valid_thread'] and b['trace_winner']==0
out={'input':'163validtasks tiednegativeK2score,93inactive sentinelthreads; constantbinary600rows yields sameobservedconfiguration score for everyparentset; notsyntheticunreachablepositive-scorecase','original':a,'corrected_left_tie':b,'source_correction':'computeKernel reduction secondcase: right<left to right<=left; deterministicleftwinner andpreservesnontiedvalues','impact':'Original invalidnegativewinner means no threadwrites D_resP; zero-memsetparents then decode asinvalidindices. No knownunsafeconstant GPUrun performed.'}
(P/'reduction_tie_witness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
