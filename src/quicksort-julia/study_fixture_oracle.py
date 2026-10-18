from pathlib import Path
import json,struct,random,hashlib,shutil
R=Path('/home/imo/HeCBench_julia');p=R/'src/quicksort-julia';f=[24];oracle=[];records=[]
for tid,fmt,bytesize in [(0,'I',4),(1,'f',4),(2,'d',8)]:
 for n,mode in [(1,0),(127,0),(256,0),(257,0),(1728,0),(4096,0),(513,1),(1024,2)]:
  a=list(range(1,n+1))
  if mode==0:random.Random(19937+n).shuffle(a)
  if mode==1:a=[42]*n
  if mode==2:a.reverse()
  words=(n*bytesize//64+1)*64//bytesize;guard=4294967295 if tid==0 else -1234567
  bits=lambda v:int.from_bytes(struct.pack('<'+fmt,v),'little')
  ab=[bits(v)for v in a]+[bits(guard)]*(words-n);dn=[bits(v)for v in sorted(a)]+[bits(guard)]*(words-n)
  f +=[tid,n,mode,words]+ab+dn;oracle+=dn
  records.append({'type':['UInt32','Float32','Float64'][tid],'n':n,'pattern':['seeded_permutation','all42','reverse'][mode],'rounded_source_payload_words':words,'logical_sorted_counts':'Each1..N once'if mode!=1 else'42exactlyNtimes','padding_bits':bits(guard)})
(p/'fixtures_u64.bin').write_bytes(struct.pack('<%dQ'%len(f),*f));(p/'oracle_sorted_u64.bin').write_bytes(struct.pack('<%dQ'%len(oracle),*oracle));(p/'independent_oracle.json').write_text(json.dumps({'cases':records,'prefix_words':len(f),'oracle_sorted_words':len(oracle),'fixture':'PythonRandom19937+Nshuffle seededpermutation1..N, all42/reverse; alltypedpositiveintegers<=4096, exactFloat32andFloat64; notoriginalC++shuffleCLIsequence','oracle':'Independentsortandfullpermutation/multiset counts, exacttypedIEEEbitencoding; typedUInt64artifactencodingpreservesoriginalrawUInt32/Float32/Float64bits','source_scratch':'HostdninitiallysortedinputasoriginalRUN_CPU_SORTS path; sourceauditactualcustomkernelsrequired, notusingthisknownanswerascomputation','source_padding':'Exactsource((sizeofT*N)/64+1)*64 roundedcopies/payload, initializedtypedcanaries; unusedpaddingnotpartsortedlogicaldata. 256Bouteralignmentguard planned.','actual_inputs_sha256':hashlib.sha256((p/'fixtures_u64.bin').read_bytes()).hexdigest()},indent=2)+'\n');print(len(f),len(oracle))
