from pathlib import Path
import random,json,hashlib
p=Path(__file__).resolve().parent
rng=random.Random(19937);n=193;M=97;K=21;a=[]
for i in range(n):
 row=[((i*17+k*13)%37-18)/8 for k in range(K)]
 if i==0:row=[32. if k==3 else -32. for k in range(K)]
 if i==192:row=[0.]*K
 a.extend(row)
rands=[rng.gauss(0,1)for _ in range(M*K)];a2=[]
for i in range(5):
 row=[0.]*K
 if i==1:row[0]=row[1]=1.;row[20]=-1.
 if i==2:row[7]=row[20]=2.
 if i==3:row[4]=row[7]=2.
 if i==4:row=[8.]*K
 a2.extend(row)
arrays={'ALPHAS1':a,'DRAWS1':rands,'ALPHAS2':a2,'DRAWS2':[0.]*(3*K)}
lines=['// Generated deterministic shared study inputs. No copyrighted external dataset.','// Python3 random.Random(19937).gauss(0,1), MT19937; values frozen as roundtrip Float64 literals.']
for name,data in arrays.items():lines.append('static const double '+name+'['+str(len(data))+']={'+','.join(format(x,'.17g')for x in data)+'};')
q=p/'study_fixture.h';q.write_text('\n'.join(lines)+'\n');provenance={'generator':'src/mcpr-julia/study_fixture_generator.py','seed':19937,'draw_generator':'Python3 random.Random.gauss mean0 sigma1; deterministiccachedBoxMuller onMT19937, distinctfromoriginalstdnormal_distributionstream','cases':[{'n':193,'K':21,'M':97,'fixture':'Dyadicalpha means, firstrowdominantcat3, lastrowzeromeans;97finiteGaussianvectorssharedacrossrows'},{'n':5,'K':21,'M':3,'fixture':'Separate deterministiczero-noise ties; expectedwinners20,0,20,4,20'}],'source_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'scope':'SeededMCfixtureandseparatetiecontrols,notoriginalCSVdata/CLIreader/defaultn26280M10000. No claimedMCdistributionerror or statisticalconvergence certificate.'};(p/'study_fixture_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
