#!/usr/bin/env python3
from pathlib import Path
import struct,json,hashlib,argparse,shutil
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--original-data',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
sets=[]
for i in range(24):
 if i==0:s=set()
 elif i==1:s=set(range(17))
 elif i in [2,4]:s=set(range(0,17,2))
 elif i==3:s=set(range(1,17,2))
 else:s={x for x in range(17)if (7*x+3*i)%(i%5+3)<i%3+1}
 sets.append(sorted(s))
records=[]
for name,ss in [('heterogeneous_full24',sets),('heterogeneous_partial19',sets[:19])]:
 elems=[x for s in ss for x in s];raw=struct.pack('<IIQ',len(ss),16,len(elems))+struct.pack('<'+'I'*len(ss),*(len(s)for s in ss))+struct.pack('<'+'I'*len(elems),*elems);f=a.output/(name+'.bin');f.write_bytes(raw);records.append({'name':name,'file':f.name,'k':len(ss),'source_universe_field':16,'effective_U':17,'partition':8,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
f=a.output/'original_uniform1000.bin';shutil.copy2(a.original_data,f);raw=f.read_bytes();records.insert(0,{'name':'original_uniform1000','file':f.name,'k':1000,'source_universe_field':1000,'effective_U':1001,'partition':1000,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'source_generator_distinct_sets':1,'all_set_cardinalities':95})
(a.output/'fixture_manifest.json').write_text(json.dumps({'benchmark':'si','cases':records,'independent_oracle':'Pythonsetintersectionforeachactualpair; noGEMM/transcription; completebinaryincidenceinputs separatelyverified','control_scope':'empty/universal/even/odd/identical/variable-cardinalitysets, zerosymbolallowedbysourcefileformat; partialtileactualoriginalwitnessplanned','original_generator':'actualCMakeoriginalbinary--iterations100--universe1000-k1000; allsetsidenticalbecauseper-setRNGseedreset; asc/descsamebytes'},indent=2)+'\n');print(json.dumps(records,indent=2))
