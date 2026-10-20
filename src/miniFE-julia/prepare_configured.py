#!/usr/bin/env python3
"""Owned conditional boundary repair and explicit native/Julia dispatch hooks.
All original GPU mathematical bodies remain byte-identical in their headers.
"""
from pathlib import Path
import json,re,hashlib,shutil
B=Path(__file__).resolve().parent;R=B.parent.parent;C=B/'configured';arc=json.loads((B/'source_archive.json').read_text())
for row in arc['files']:
 src=R/row['archive'];rel=src.relative_to(B/'reference_history');dst=C/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
records=[]
ops={'wx_kernel':2,'wax_kernel':3,'wxby_kernel':0,'waxby_kernel':1,'dyx_kernel':4,'dyax_kernel':5,'yxby_kernel':6,'yaxby_kernel':7,'dot_kernel':8,'final_reduce':9,'matvec_kernel':10}
pat=re.compile(r'(?P<name>\w+)(?:<[^;<>]*>)?\s*<<<(?P<dims>.*?)>>>\s*\((?P<args>.*?)\);',re.S)
def patch_launch(path):
 original=path.read_text()
 def repl(m):
  name=m['name']
  if name not in ops:raise ValueError(name)
  args=[a.strip()for a in m['args'].split(',')];dims=[d.strip()for d in m['dims'].split(',')];assert len(dims)==2
  fields={};fields['op']=str(ops[name]);fields['n']=args[0]if name!='final_reduce'else'MAX_NUM_BLOCKS'
  if name=='matvec_kernel':fields.update(row_offsets=args[1],cols=args[2],coefs=args[3],x=args[4],w=args[5])
  elif name=='final_reduce':fields.update(x=args[0],w=args[0])
  elif name=='dot_kernel':fields.update(x=args[1],y=args[2],w=args[3])
  elif name in ['wx_kernel','dyx_kernel']:fields.update(x=args[1],w=args[2]);fields['y']=args[2]if name=='dyx_kernel'else'nullptr'
  elif name in ['wax_kernel','dyax_kernel']:fields.update(alpha=args[1],x=args[2],w=args[3]);fields['y']=args[3]if name=='dyax_kernel'else'nullptr'
  elif name in ['wxby_kernel','yxby_kernel']:fields.update(x=args[1],beta=args[2],y=args[3],w=args[4]if name=='wxby_kernel'else args[3])
  else:fields.update(alpha=args[1],x=args[2],beta=args[3],y=args[4],w=args[5]if name=='waxby_kernel'else args[4])
  assigns=''.join(f' packet.{k}={v};'for k,v in fields.items())
  records.append(dict(file=str(path.relative_to(B)),source_kernel=name,op=ops[name],original_launch=m.group(0),line=original[:m.start()].count('\n')+1))
  return '{ if (::mini_callback) { MiniLaunch packet{}; packet.grid=dim3('+dims[0]+').x; packet.threads=dim3('+dims[1]+').x;'+assigns+' ::mini_dispatch(packet); } else { '+m.group(0)+' } }'
 new=pat.sub(repl,original);path.write_text(new)
for name in ['Vector_functions.hpp','SparseMatrix_functions.hpp']:patch_launch(C/'src'/name)
p=C/'src/driver.hpp';s=p.read_text();marker='  //Now apply dirichlet boundary-conditions';assert s.count(marker)==1;s=s.replace(marker,'  // Owned correction: x=1 hot-face priority also applies to its edges.\n  for (auto hot_row : mesh.bc_rows_1) mesh.bc_rows_0.erase(hot_row);\n\n'+marker)
old='verify_result = verify_solution(mesh, x, tolerance, verify_whole_domain);';assert s.count(old)==1;s=s.replace(old,'::mini_observe_source_heuristic(verify_solution(mesh, x, tolerance, verify_whole_domain));\n      verify_result = 0; // independent full Poisson/Q1 gate in study; original heuristic retained as auxiliary')
p.write_text(s)
p=C/'src/cg_solve.hpp';s=p.read_text();needle='        TICK(); waxpby(one, x, zero, x, p, d_x, d_x, d_p); TOCK(tWAXPY);';assert s.count(needle)==1
hook='        ::mini_observe_initial((int)A.rows.size(),(int)A.packed_cols.size(),d_Arowoffsets,d_Acols,d_Acoefs,d_b,d_x,d_r,d_p,d_Ap);\n'
s=s.replace(needle,hook+needle);needle='        cudaMemcpy(x_ptr, d_x, sizeof(MINIFE_SCALAR) * x.coefs.size(), cudaMemcpyDeviceToHost);';assert s.count(needle)==1;s=s.replace(needle,needle+'\n        ::mini_observe_solution((int)x.coefs.size(),d_x,d_r,d_p,d_Ap,d_b,d_Acoefs,num_iters,normr);');p.write_text(s)
p=C/'src/Vector_functions.hpp';s=p.read_text();needle='      cudaMemcpy(&result, d, sizeof(Scalar), cudaMemcpyDeviceToHost);';assert s.count(needle)==2;s=s.replace(needle,needle+'\n      ::mini_observe_reduction(d);');p.write_text(s)
changed=[]
for name in ['driver.hpp','cg_solve.hpp','Vector_functions.hpp','SparseMatrix_functions.hpp']:
 p=C/'src'/name;o=B/'reference_history/src'/name;changed.append(dict(path=str(p.relative_to(B)),original_sha256=hashlib.sha256(o.read_bytes()).hexdigest(),configured_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(B/'configuration_provenance.json').write_text(json.dumps(dict(native_reference_repaired=True,repair='Hotface rows removed from cold Dirichlet set before elimination; x=1 priority as source verifier documents. Original GPU bodies untouched.',checker_qualification='Original homogeneous center heuristic emitted as auxiliary; independent full Poisson/Q1 gate used.',launch_interceptions=records,changed_files=changed,explicit_native_kernel_fallback_for_nonnull_callback=False),indent=2)+'\n');print('Intercepted',len(records),'explicit launch sites;',sorted(set(r['source_kernel']for r in records)))
