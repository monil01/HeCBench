"""Locate source-grounded cross-lane packed-word hazard; no GPU execution."""
from pathlib import Path
import json
P=Path(__file__).resolve().parents[1]
s=(P/'generate_study_fixture.py').read_text().split('reports=[]')[0]
def packed_witness(entries,active):
 examples=[]
 for phase,part in [('decrement',(1,3)),('increment',(3,5))]:
  owners={}
  for lane,e in enumerate(entries):
   if e not in active:continue
   for byte in e[part[0]:part[1]]:owners.setdefault(byte//4,[]).append({'lane_zero_based':lane,'byte_index_zero_based':byte})
  for word,users in owners.items():
   if len(set(a['lane_zero_based']for a in users))>1:examples.append({'phase':phase,'word_index_zero_based':word,'concurrent_byte_writers':users})
 return {'cross_lane_packed_word_witnesses':examples}
ns={'__file__':str(P/'generate_study_fixture.py'),'packed_witness':packed_witness};exec(s.replace('waves.append(wave)','wave.update(packed_witness(entries,active));waves.append(wave)'),ns)
r=ns['simulate'](1024,5492)[0];print(json.dumps({'diagnostic_only':True,'is_observed_numerical_failure':False,'canonical_primary_seed':5492,'report':r,'interpretation':'Distinct concurrent native byte stores may touch one UInt32 word. Historical wholeword read-modify-write creates an additional cross-byte race risk; actual historical canonical GPU diagnostic happened to pass.'},indent=2))
