#!/usr/bin/env python3
"""Bounded sparse source-generator fixture; no dense tensor or GPU allocation."""
import pathlib,random,importlib.util,argparse,json,hashlib,itertools,shutil,sys,resource,math
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=pathlib.Path,required=True);p.add_argument('--seed',type=int,default=20261020);p.add_argument('--skip-full',action='store_true');a=p.parse_args();B=pathlib.Path(__file__).resolve().parent;D=a.output_dir.resolve();D.mkdir(parents=True,exist_ok=True)
# Original generator selects <=82,82,82,11 coordinates per level; <=6,065,048
# sparse records, conservatively <=243MB text. Enforce <=300MiB per file even
# if source behavior differs, and reserve >2GiB beyond that bound before writing.
limit=300*1024*1024;free=shutil.disk_usage(D).free
if free<2*1024**3+limit:raise RuntimeError('Insufficient free space for bounded sparse fixture plus2GiBreserve')
resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
assert math.prod([math.ceil(.02*n)for n in [4096,4096,4096,512]])==6065048
shutil.copy2(B/'reference_history/toy.tns',D/'toy.tns')
def tensor(name,dims,selected):
 with (D/name).open('w')as f:
  f.write(str(len(dims))+'\n'+' '.join(map(str,dims))+'\n')
  for coords in itertools.product(*(range(n)for n in dims)):
   if selected(coords):
    key=sum((j+3)*x for j,x in enumerate(coords));v=((key*13)%31-15)/16
    if v==0:v=.0625
    f.write(' '.join(str(x+1)for x in coords)+' '+str(v)+'\n')
tensor('control3D.tns',(5,7,9),lambda v:(v[0]*17+v[1]*11+v[2]*7)%9<4)
tensor('longFiber4D.tns',(3,4,17,23),lambda v:True)
if not a.skip_full:
 full=D/'sourceSize4D.tns'
 if full.exists():raise FileExistsError('Refuse overwriting existing full fixture: '+str(full))
 spec=importlib.util.spec_from_file_location('original_tensor_generator',B/'reference_history/generate_tensor.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);random.seed(a.seed);status=mod.main(['generate_tensor.py',str(full),'2%4096','2%4096','2%4096','2%512']);assert status==0
rows=[]
for f in sorted(D.glob('*.tns')):
 h=hashlib.sha256();n=0
 with f.open('rb')as stream:
  for line in stream:h.update(line);n+=1
 rows.append({'name':f.name,'path':str(f),'bytes':f.stat().st_size,'sha256':h.hexdigest(),'nonzeros':n-2})
out={'benchmark':'mmcsf','full_generator':'reference_history/generate_tensor.py','full_dimensions':[4096,4096,4096,512],'full_rates_percent':[2,2,2,2],'seed':a.seed,'Python':sys.version,'hard_perfile_limit_bytes':limit,'free_before_bytes':free,'maximum_sparse_records':6065048,'original_generator_source_sha256':hashlib.sha256((B/'reference_history/generate_tensor.py').read_bytes()).hexdigest(),'fixtures':rows,'scope':'SourceMakefilegenerator dimensions/rates with explicit deterministicstudyseed, not original archivedunseededrealization. Toy originalbytes; smallcontrols authorgeneratedbinaryfractions. Sparse only, hard300MiB filelimit and2GiBreserve.'};q=D/'manifest.tmp';q.write_text(json.dumps(out,indent=2)+'\n');q.replace(D/'manifest.json');print(json.dumps(out))
