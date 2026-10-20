#!/usr/bin/env python3
"""Named synthetic unit-sphere catalogs; original sky data are NOT present.
Independent exact-integer suffix-multiplicity pair oracle from source input types.
Decimal dots certify bin classification away from all actual FP64 cutpoints.
"""
from pathlib import Path
from decimal import Decimal,localcontext
import math,struct,json,hashlib,array
P=Path(__file__).resolve().parent;D=P/'data';G=16384
bounds=struct.unpack('<63d',(D/'source_boundaries63.bin').read_bytes())
vec=[]
for bi,base in enumerate([0,20,40,65,90,110,130,150]):
 for off in [0,.001,.01,.05,.2,1,3]:
  ra=math.radians(base+off);dec=math.radians((bi%3-1)*.4+off*.01);vec.append((math.cos(ra)*math.cos(dec),math.sin(ra)*math.cos(dec),math.sin(dec)))
T=len(vec);matrix=[];mindist=math.inf;minratio=math.inf
with localcontext() as c:
 c.prec=90;u=Decimal(2)**-53;gamma=5*u/(1-5*u)
 for a in vec:
  row=[]
  for b in vec:
   prods=[Decimal(x)*Decimal(y)for x,y in zip(a,b)];dot=sum(prods);budget=gamma*sum(abs(v)for v in prods);dist=min(abs(dot-Decimal(v))for v in bounds[:31]);mindist=min(mindist,float(dist));minratio=min(minratio,float(dist/budget)if budget else math.inf)
   assert dist>budget*10,(a,b,dist,budget)
   bin=next((k+1 for k in range(30,-1,-1) if dot<Decimal(bounds[k])),0);row.append(bin)
  matrix.append(row)

def digest(b):return hashlib.sha256(b).hexdigest()
def jk_sizes(n,j):q=n//j;q-=q%4;return [q]*(j-1)+[n-(j-1)*q]
def groups(n,j):
 out=[]
 for g,s in enumerate(jk_sizes(n,j)):out.extend([g]*s)
 return out

def physical_auto(ids,j):
 hist=[[0]*64 for _ in range(j)];suffix=[0]*T
 for t in ids:suffix[t]+=1
 for i,t in enumerate(ids):
  suffix[t]-=1;out=hist[groups_cache[i]]
  for other,count in enumerate(suffix):out[matrix[t][other]]+=count
  # Actual upper-triangle sentinel assignments survive source histogram output.
  out[31]+=(i%G)+1
 return hist

def cross(data,random,j):
 dh=[[0]*T for _ in range(j)];rh=[0]*T
 for g,t in zip(groups(len(data),j),data):dh[g][t]+=1
 for t in random:rh[t]+=1
 out=[[0]*64 for _ in range(j)]
 for g in range(j):
  for t,n in enumerate(dh[g]):
   for other,m in enumerate(rh):out[g][matrix[t][other]]+=n*m
 return out
records=[]
for ci,(n,nr,njk) in enumerate([(97176,2,10),(16388,1,4),(8,1,2)]):
 ids=[[(i*17+ci*3)%T for i in range(n)]]+[[((i*19+r*13+ci*5)%T)for i in range(n)]for r in range(1,nr+1)]
 paths=[]
 for k,cat in enumerate(ids):
  f=D/f'case{ci}_catalog{k}.txt';paths.append(f)
  with f.open('w')as out:
   for t in cat:out.write('0 0 '+' '.join(format(x,'.17g')for x in vec[t])+'\n')
 groups_cache=groups(n,njk);dd=physical_auto(ids[0],njk);dr=[cross(ids[0],r,njk)for r in ids[1:]]
 groups_cache=[0]*n;rr=[physical_auto(r,1)[0]for r in ids[1:]]
 npad=((n+G-1)//G)*G;nk=npad//G
 def frames(k,tile):
  chunk=ids[k][tile*G:(tile+1)*G]
  return [struct.pack('<'+str(G)+'d',*([vec[t][axis]for t in chunk]+[0.]*(G-len(chunk))))for axis in range(3)]
 sequence=[]
 def add(k,tile):sequence.extend(frames(k,tile))
 def auto(k):
  for i in range(nk):
   add(k,i)
   for z in range(i+1,nk):add(k,z)
 auto(0)
 for k in range(1,nr+1):
  for i in range(nk):
   add(0,i)
   for z in range(nk):add(k,z)
  auto(k)
 unique={}
 for blob in sequence:unique.setdefault(digest(blob),blob)
 # Compact lossless declaration: each unique actual-upload expected frame is a file,
 # every repeated operation is bound to its position in the complete sequence.
 shared=D/'input_frames';shared.mkdir(exist_ok=True)
 for h,blob in unique.items():
  f=shared/(h+'.bin')
  if not f.exists():f.write_bytes(blob)
 prefix=4+njk+63+sum(len(b)//8 for b in unique.values())
 f=D/f'case{ci}_oracle.bin';flat=[v for row in dd for v in row]+[v for cat in dr for row in cat for v in row]+[v for cat in rr for v in cat]
 f.write_bytes(struct.pack('<'+str(len(flat))+'q',*flat))
 meta={'case':ci,'n':n,'nr':nr,'njk':njk,'n_effective_source_requested':97178 if ci==0 else n,'n_padded':npad,'jk_sizes':jk_sizes(n,njk),'catalogs':[str(f.relative_to(P))for f in paths],'catalog_sha256':[digest(f.read_bytes())for f in paths],'oracle':str(f.relative_to(P)),'oracle_sha256':digest(f.read_bytes()),'source_H2D_frame_sequence':[digest(b)for b in sequence],'unique_H2D_frame_firstseen':list(unique),'prefix_count':prefix,'input_H2D_uploads':len(sequence),'auto_ACF_calls_each':nk*(nk+1)//2,'cross_ACF_calls_each':nk*nk,'exact_total_real_auto_pairs':n*(n-1)//2,'exact_total_real_cross_pairs':n*n,'raw_histogram_values':len(flat)}
 (D/f'case{ci}.json').write_text(json.dumps(meta,indent=2)+'\n');records.append(meta)
manifest={'fixture_kind':'Declaredsyntheticrepeateddirectioncatalogs; notoriginalmissingcosmologicaldataset','source_boundary_export':'export_boundaries.cpp','source_boundary_sha256':digest((D/'source_boundaries63.bin').read_bytes()),'coordinate_types':vec,'dot_bin_matrix':matrix,'independent_bin_rule':'Decimal90digitexactIEEE64inputproducts; descendingstrictcomparisonsagainstactualexportedsourcecutpoints','min_distance_to_source_cutpoint':mindist,'minimum_distance_over_gamma5_fp64_dot_rounding_bound':minratio,'FMA_or_separate_dot_rounding':'Allpairs fartherthan10*gamma5*sumabsproducts; exact bin classifications independent of FP64 contraction','oracle_method':'Exactinteger suffix-multiplicity pair counting preserving everycatalogposition andsourcegroup-assignment; actualsourceuppertrianglepadding sentinel31 countedseparately. FullDD/DR/RR0..30 aretruephysicalpaircounts aftercorrectpaddingselector; sourceone-sidedJKreportsnottextbookjackknife.','cases':records,'prefix_count_total':sum(x['prefix_count']for x in records),'native_reference_repaired':True,'exclude_untouched_native_aggregates':True};(D/'fixture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'cases':[(x['n'],x['nr'],x['njk'],x['input_H2D_uploads'],x['prefix_count'])for x in records],'prefix':manifest['prefix_count_total'],'unique_frames':len(list((D/'input_frames').glob('*.bin'))),'margin_ratio':minratio}))
