from pathlib import Path
import shutil,json,hashlib,difflib
O=Path(__file__).parent;R=Path('/home/imo/HeCBench_julia');C=Path('/tmp/roofline-completion-20261021/current537_display_traits_consumer');sha=lambda b:hashlib.sha256(b).hexdigest();D=O/'complete_combined'
# Preserve prior packet; new complete proposal uses byte-copied current consumer bases.
for group in ['base','proposed']:
 for src in (C/group).rglob('*'):
  if src.is_file():
   dst=D/group/src.relative_to(C/group);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
loader=D/'proposed/ontology/tools/current537_display_traits.py';s=loader.read_text().replace('import json\n','import json\nimport os\nimport stat\n',1).replace('def __init__(self, root, population, review_sha256=None):','def __init__(self, root, population, review_sha256=None, metadata_lock_fd=None):',1).replace('        self.population = population','        self.population = population\n        self.borrowed_lock_fd = metadata_lock_fd',1)
a="""            def __enter__(inner):
                inner.file = open('/tmp/hecbench-roofline-metadata.lock','r')
                fcntl.flock(inner.file,fcntl.LOCK_SH)
            def __exit__(inner,*unused):
                fcntl.flock(inner.file,fcntl.LOCK_UN)
                inner.file.close()"""
b="""            def __enter__(inner):
                if self.borrowed_lock_fd is not None:
                    # Borrow the exact inherited open-file description. EX -> EX
                    # never downgrades the parent's lease; NB never waits on it.
                    actual = os.fstat(self.borrowed_lock_fd)
                    expected = os.stat('/tmp/hecbench-roofline-metadata.lock')
                    if (not stat.S_ISREG(actual.st_mode)
                            or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino)):
                        raise ValueError('Borrowed FD is not canonical metadata lock')
                    fcntl.flock(self.borrowed_lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    inner.file = None
                else:
                    inner.file = open('/tmp/hecbench-roofline-metadata.lock','r')
                    fcntl.flock(inner.file,fcntl.LOCK_SH)
            def __exit__(inner,*unused):
                # Never unlock/close the borrowed lease: parent still owns it.
                if inner.file is not None:
                    fcntl.flock(inner.file,fcntl.LOCK_UN)
                    inner.file.close()"""
assert a in s;s=s.replace(a,b,1).replace("return {'authority':self.authority,'issues':self.issues", "return {'metadata_lock_protocol':'borrowed_verified_EX_unchanged'if self.borrowed_lock_fd is not None else 'owned_SH',\n                'authority':self.authority,'issues':self.issues",1);loader.write_text(s)
for f in ['ontology/tools/generate_ontology_report.py','ontology/tools/generate_relationship_graph.py']:
 p=D/'proposed'/f;s=p.read_text();s=s.replace('current537_display_review_sha256=None):','current537_display_review_sha256=None,metadata_lock_fd=None):',1).replace('traits=DisplayTraits(root,population,current537_display_review_sha256)','traits=DisplayTraits(root,population,current537_display_review_sha256,metadata_lock_fd)',1)
 anchor="    args=parser.parse_args()";assert anchor in s;s=s.replace(anchor,"    parser.add_argument('--metadata-lock-fd',type=int,help='Borrowed canonical metadata EX-lock FD inherited via pass_fds; never downgraded, unlocked or closed.')\n"+anchor,1)
 if 'ontology_report' in f:s=s.replace('build(root,args.population,args.current537_display_review_sha256)','build(root,args.population,args.current537_display_review_sha256,args.metadata_lock_fd)',1)
 else:s=s.replace('build(root,args.neighbors,args.population,args.current537_display_review_sha256)','build(root,args.neighbors,args.population,args.current537_display_review_sha256,args.metadata_lock_fd)',1)
 p.write_text(s)
def refresh_fd(s):
 s=s.replace('    def run(label,argv):','    def run(label,argv,*,pass_fds=()):',1).replace('subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True)','subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,pass_fds=pass_fds)',1)
 a="            if args.current537_display_review_sha256:report_command+=['--current537-display-review-sha256',args.current537_display_review_sha256]\n            if run('ontology_report',report_command):"
 b="""            report_pass_fds=()
            if args.current537_display_review_sha256:
                report_command+=['--current537-display-review-sha256',args.current537_display_review_sha256,
                                 '--metadata-lock-fd',str(lock.fileno())]
                report_pass_fds=(lock.fileno(),)
            if run('ontology_report',report_command,pass_fds=report_pass_fds):"""
 assert a in s;s=s.replace(a,b,1);return s
f='tools/v100_roofline_refresh.py';base=(O/'base'/f).read_text();candidate=refresh_fd((O/'proposed_flag_only'/f).read_text());future=refresh_fd((O/'composed_flag_only'/f).read_text())
for group,text in [('base',base),('proposed',candidate)]:
 p=D/group/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
p=O/'future_method_and_traits_FD'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(future)
for name,a,b in [('refresh_traits_FD_only.patch',base,candidate),('future_method_traits_FD_vs_live.patch',base,future),('future_method_traits_FD_vs_method_v2.patch',(O/'future_method_base'/f).read_text(),future)]:
 (O/name).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+f,tofile='b/'+f)))
review=json.loads((C/'root_review.template.json').read_text());review['loader']['sha256']=sha(loader.read_bytes());review['limitations'].append('Approved loader includes borrowed-EX-FD protocol; parent must pass exact held canonical FD via pass_fds and explicit --metadata-lock-fd. Owned standalone SH remains separate.');(D/'root_review.template.json').write_text(json.dumps(review,indent=2)+'\n')
patch=[];bindings={}
for p in sorted((D/'proposed').rglob('*')):
 if not p.is_file():continue
 f=str(p.relative_to(D/'proposed'));b=D/'base'/f;old=b.read_text().splitlines(True)if b.exists()else[];text=p.read_text();bindings[f]={'proposed_sha256':sha(p.read_bytes()),**({'base_sha256':sha(b.read_bytes())}if b.exists()else{'new_file':True})};patch.extend(difflib.unified_diff(old,text.splitlines(True),fromfile='a/'+f if b.exists()else'/dev/null',tofile='b/'+f))
(D/'minimal.patch').write_text(''.join(patch));(D/'authoring_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');(O/'composition_bindings.json').write_text(json.dumps({'live_base_sha256':sha(base.encode()),'combined_refresh_sha256':sha(candidate.encode()),'future_method_v2_base_sha256':sha((O/'future_method_base'/f).read_bytes())if False else sha((O/'future_method_base/tools/v100_roofline_refresh.py').read_bytes()),'future_method_traits_FD_refresh_sha256':sha(future.encode()),'new_loader_sha256':sha(loader.read_bytes()),'overlay_sha256':review['overlay']['sha256'],'prior_consumer_packet_untouched':str(C),'future_method_status':'UNADOPTED_PENDING_ACTUAL_CALIBRATION002_AND_ROOT_REVIEW','actual_root_review_sha256':None},indent=2)+'\n');print(json.dumps({'new_loader_sha256':sha(loader.read_bytes()),'combined_refresh_sha256':sha(candidate.encode()),'future_composed_refresh_sha256':sha(future.encode()),'combined_patch_sha256':sha((D/'minimal.patch').read_bytes())}))
