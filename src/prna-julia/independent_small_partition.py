#!/usr/bin/env python3
"""Independent explicit enumeration of two potential nested pairs and exterior microstates.
No source dynamic-program recurrence is called. Parameters are actual serialized Float32 inputs.
"""
import struct,json,math,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]);layout=json.loads((p/'parameter_layout.json').read_text());raw=(p/'params_u32.bin').read_bytes();F=layout['fields']
def f(name,idx=(),base=5):
 off=F[name]['offset_bytes'];index=0
 for i in idx:index=index*base+i
 return struct.unpack_from('<f',raw,off+index*4)[0]
def special(seq):
 count=struct.unpack_from('<i',raw,F['ntriloop']['offset_bytes'])[0];off=F['triloop']['offset_bytes']
 for k in range(count):
  if struct.unpack_from('<5i',raw,off+24*k)==tuple(seq):return struct.unpack_from('<f',raw,off+24*k+20)[0]
 return None
# X,A,C,G,U = 0,1,2,3,4. GCAAAGC has outer(0,6), inner(1,5).
s=[3,2,1,1,1,3,2];inner=special(s[1:6]);inner=f('hairpin_loop_initiation',(3,)) if inner is None else inner
outer=f('hairpin_loop_initiation',(5,))+f('tstackh',(3,2,2,3));stack=f('stack',(3,2,2,3));d5=f('dangle_5p',(3,2,3));d3=f('dangle_3p',(3,2,2));tm=f('tstackm',(3,2,2,3))
weights={'unpaired':1.0,'outer_hairpin':math.exp(-outer),'both_nested_stack':math.exp(-(stack+inner)),'inner_no_dangle':math.exp(-inner),'inner_left_dangle':math.exp(-(inner+d5)),'inner_right_dangle':math.exp(-(inner+d3)),'inner_terminal_stack':math.exp(-(inner+tm))};Z=math.fsum(weights.values());outerP=(weights['outer_hairpin']+weights['both_nested_stack'])/Z;innerP=(Z-1-weights['outer_hairpin'])/Z
# Lowercase inner pair is forbidden by source uppercase pairing constraints;
# potential-neighbor mask still permits the uppercase outer pair.
Zconstrained=1+math.exp(-outer)
r={'kind':'independent_explicit_boltzmann_microstate_enumeration','sequence':'GCAAAGC','parameters_sha256':hashlib.sha256(raw).hexdigest(),'parameter_values':{'inner_hairpin':inner,'outer_hairpin':outer,'stack':stack,'left_dangle':d5,'right_dangle':d3,'terminal_stack':tm},'weights':weights,'partition_function':Z,'free_energy_neg_log_partition':-math.log(Z),'pair_probabilities':{'0,6':outerP,'1,5':innerP},'constrained_sequence':'GcAAAgC','constrained_free_energy':-math.log(Zconstrained),'constrained_outer_probability':math.exp(-outer)/Zconstrained,'all_A_control':{'partition_function':1,'free_energy':0,'probabilities':0},'scope':'Exact enumerated source-energy microstates for2possiblepairs atn7, not full292biologicalgroundtruth; table inputenergies evaluatedFloat64 independently fromFloat32 recurrence rounding. Source isolatedpotentialpair mask/exteriorfourmicrostate convention explicit.'};(p/'independent_small_partition.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

(p/"tiny_oracle_f64.bin").write_bytes(struct.pack("<5d",r["free_energy_neg_log_partition"],r["pair_probabilities"]["0,6"],r["pair_probabilities"]["1,5"],r["constrained_free_energy"],r["constrained_outer_probability"]))
