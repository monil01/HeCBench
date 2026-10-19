"""Independent exhaustive nested secondary-structure energy certificates.
No source DP implementation is imported. Controls admit at most one nested helix
chain; the 77-base biological sequence is deliberately outside this enumeration.
"""
from pathlib import Path
from functools import lru_cache
import argparse,json,struct,math,hashlib
p=argparse.ArgumentParser();p.add_argument('--fixture-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=a.fixture_root;raw=(D/'params.bin').read_bytes();words=struct.unpack('<'+'i'*(len(raw)//4),raw);layout=json.loads((D/'parameter_layout.json').read_text())['fields'];M=json.loads((D/'fixture_metadata.json').read_text());base={'A':0,'C':1,'G':2,'U':3};cp=lambda x,y:(x,y)in[(0,3),(3,0),(1,2),(2,1),(2,3),(3,2)]
def val(name,*idx):
 offset=layout[name]['word_offset'];flat=0
 for v in idx:flat=4*flat+v
 return words[offset+flat]
def scalar(name):return val(name)
def short_loop(seq):
 for d,name,num,width in [(3,'triloop','ntriloop',5),(4,'tloop','ntloop',6),(6,'hexaloop','nhexaloop',8)]:
  if len(seq)!=width:continue
  o=layout[name]['word_offset']
  for k in range(scalar(num)):
   start=o+k*(width+1)
   if tuple(seq)==words[start:start+width]:return words[start+width]
 return None
prelog=struct.unpack_from('<f',raw,4*layout['prelog']['word_offset'])[0];reports=[]
for case in M['cases'][1:]:
 text=case['sequence'];s=tuple(base[x]for x in text);n=len(s)
 def allowed(i,j):
  pair=lambda x,y:0<=x<y<n and y-x-1>=3 and cp(s[x],s[y])
  return pair(i,j)and(pair(i-1,j+1)or pair(i+1,j-1))
 @lru_cache(None)
 def structures(i,j):
  if i>j:return ((),)
  options=list(structures(i+1,j))
  for k in range(i+4,j+1):
   if not allowed(i,k):continue
   for inner in structures(i+1,k-1):
    for rest in structures(k+1,j):options.append(((i,k),)+inner+rest)
  return tuple(options)
 terminal=lambda i,j:scalar('terminal_AU_penalty')if s[i]==3 or s[j]==3 else 0
 def hairpin(i,j):
  d=j-i-1;special=short_loop(s[i:j+1])
  if special is not None:return special
  if d<=30:e=words[layout['hairpin_loop_initiation']['word_offset']+d]
  else:e=int(words[layout['hairpin_loop_initiation']['word_offset']+30]+prelog*math.log(d/30))
  if d==3:
   e+=terminal(i,j)
   if all(x==1 for x in s[i+1:j]):e+=scalar('c_hairpin_of_3')
  else:
   e+=val('tstackh',s[i],s[j],s[i+1],s[j-1])
   if all(x==1 for x in s[i+1:j]):e+=scalar('c_hairpin_slope')*d+scalar('c_hairpin_intercept')
  if s[i]==2 and s[j]==3 and i>=2 and s[i-2:i]==(2,2):e+=scalar('bonus_for_GGG_hairpin')
  return e
 def internal(outer,inner):
  i,j=outer;ip,jp=inner;d1=ip-i-1;d2=j-jp-1
  if d1==d2==0:return val('stack',s[i],s[j],s[ip],s[jp])
  if d1==0 or d2==0:
   e=words[layout['bulge_loop_initiation']['word_offset']+d1+d2]
   if d1+d2==1:
    e+=val('stack',s[i],s[j],s[ip],s[jp]);k=i+1 if d1 else j-1;b=s[k];left=k;right=k
    while left>0 and s[left-1]==b:left-=1
    while right+1<n and s[right+1]==b:right+=1
    degeneracy=right-left+1;e+=int(-1.987213e-3*310.15*100*math.log(degeneracy))
    if b==1 and ((d1 and(s[i]==1 or s[i+2]==1))or(d2 and(s[j]==1 or s[j-2]==1))):e+=scalar('Bonus_for_Single_C_bulges_adjacent_to_C')
   else:e+=terminal(i,j)+terminal(ip,jp)
   return e
  if d1==d2==1:return val('int11',s[i],s[i+1],s[ip],s[jp],s[j-1],s[j])
  raise AssertionError('Control exposes unimplemented independent energy motif')
 candidates=[]
 for pairs in structures(0,n-1):
  if not pairs:candidates.append({'pairs':[],'energy':0});continue
  chain=sorted(pairs)
  assert all(chain[k][0]<chain[k+1][0]<chain[k+1][1]<chain[k][1]for k in range(len(chain)-1)), 'These fixtures mustnotcontainmultibranch/disjointhelices'
  e=hairpin(*chain[-1])+sum(internal(chain[k],chain[k+1])for k in range(len(chain)-1));i,j=chain[0];outer=terminal(i,j);options=[outer]
  if i>0:options.append(outer+val('dangle_5p',s[j],s[i],s[i-1]))
  if j+1<n:options.append(outer+val('dangle_3p',s[j],s[i],s[j+1]))
  if i>0 and j+1<n:options.append(outer+val('tstack',s[j],s[i],s[j+1],s[i-1]))
  candidates.append({'pairs':chain,'energy':e+min(options)})
 best=min(c['energy']for c in candidates);expected=case['MFE_scaled_Int32'];assert best==expected,(text,best,expected,candidates)
 reports.append({'sequence':text,'n':n,'all_admissible_noncrossing_structures':len(candidates),'global_optimum_Int32':best,'serial_source_MFE_Int32':expected,'minimizers':[c for c in candidates if c['energy']==best],'all_pass':True})
report={'all_pass':True,'kind':'Independent exhaustive secondary-structure enumeration, not importedsourceDP','cases':reports,'eligibility':'Controls haveonepossiblynestedhelixchain, no pseudoknots ormultiplehelixbranches. Sourcecan_pair potentialneighbor/minloop filterexplicitly retained; notunrestrictedbiologicalfolding.','energy_model':'ActualhashboundRNAthermodynamictables; independenthairpin/stack/smallinternal/bulge-degeneracy andexternaldangle/mismatch energy sum. Completeadmissiblestructureenumeration, notdynamicprogramming recurrence replay.','actual_parameter_sha256':hashlib.sha256(raw).hexdigest(),'table_coefficient_scope':'CorrectedprelogFloat32107.9000015258789 fromraw1.079×100. Longhairpin independentunitwitness separatelyconfirms.','limits':['Full77-base case receives serial-source crosscheck plusfullmatrix/prefixsuffix/terminalvalidity checks, notthisexhaustiveglobal certificate','No pseudoknot/biologicalgroundtruth/fullRNAstructure-model equivalence claim']};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
