from pathlib import Path
import random,struct,json,hashlib
p=Path(__file__).resolve().parent;r=random.Random(20261018);prefix=[5.];gold=[];cases=[];rep=3
for n,o,v in [(1,7,5),(511,7,33),(8192,32,8),(8193,32,64),(16384,1,65)]:
 idx=[r.randrange(o) for _ in range(n)];idx[0]=0;idx[-1]=o-1;units=[r.randrange(-8,9) for _ in range(n*v)];init=[r.randrange(-16,17) for _ in range(o*v)];sums=[0]*(o*v);ab=[0]*(o*v)
 for i in range(n):
  for j in range(v):k=idx[i]*v+j;sums[k]+=units[i*v+j];ab[k]+=abs(units[i*v+j])
 bound=max(abs(a)+rep*b for a,b in zip(init,ab));assert bound<2**24
 prefix.extend([float(n),float(o),float(v)]);prefix.extend(map(float,idx));prefix.extend(x/16 for x in units);prefix.extend(x/16 for x in init);expected=[(a+rep*b)/16 for a,b in zip(init,sums)]+[-8.]*17
 for mode in [1,2]:gold.extend(expected)
 cases.append({'batch':n,'output_size':o,'vector_dim':v,'blocks':n//8192+1,'kernel_modes':[1,2],'initialization':'Nonzero signed dyadic output/16','contributions':'Signed randomintegers[-8,8]/16, allindicesvalid0..O-1','max_absolute_partial_units_bound':bound,'bound_implies_exact_FP32_atomic_order_independence':True,'shared_bytes':4*(o*v+512),'source_global_atomic_updates_per_launch':(n//8192+1)*o*v})
raw=struct.pack('<'+str(len(prefix))+'f',*prefix);graw=struct.pack('<'+str(len(gold))+'d',*gold);(p/'fixture.bin').write_bytes(raw);(p/'expected_outputs.bin').write_bytes(graw);meta={'seed':20261018,'internal_repeats':rep,'cases':cases,'prefix_count':len(prefix),'output_count':len(gold),'fixture_sha256':hashlib.sha256(raw).hexdigest(),'gold_sha256':hashlib.sha256(graw).hexdigest(),'oracle':'Independentintegerhistogram ofeachindexedvectorcomponent; expectedunitsinit+3*sum(sourceunits), divide16exact. Absolute-partial-unitbound<2^24 coversany shared/globalatomicorder andallrepeatedlaunches.','scope':'Bothoriginalnativekernels, original8192chunks/512threads/floorN8192+1incl emptyblocks, 5namedshapes notdefault10millionbatch CLI. Non-dyadicarbitraryFP32rounding/invalidindicesunverified.'};(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:meta[k] for k in ['prefix_count','output_count']})
