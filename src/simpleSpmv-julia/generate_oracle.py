from pathlib import Path
import struct,json,hashlib,math,collections,array
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();off=0
def read(fmt,n=1):
 global off
 out=struct.unpack_from('<'+fmt*n,raw,off);off+=struct.calcsize(fmt)*n;return out
nc,=read('I');expected=[];desc=[];prefix=0
for z in range(nc):
 n,nz,tpr=read('I',3);mat=read('f',n*n);x=read('f',n);row=read('Q',n+1);col=read('Q',nz);val=read('f',nz);assert row[0]==0 and row[-1]==nz
 y=[];lengths=[]
 for r in range(n):
  js=range(row[r],row[r+1]);lengths.append(row[r+1]-row[r]);assert all(mat[r*n+col[k]]==val[k]for k in js)
  sparse=math.fsum(val[k]*x[col[k]]for k in js);dense=math.fsum(mat[r*n+c]*x[c]for c in range(n));assert sparse==dense;y.append(sparse)
 for bs in [32,64,128,256,512,1024]:
  assert n%(bs//tpr)==0
  for method in ['dense','csr','vector_csr']:expected.extend(y+[16777215.]*16)
 prefix+=n*n+n+(n+1)+nz+nz
 desc.append({'case':z,'rows':n,'nnz':nz,'mean_nnz_per_row':nz/n,'source_prevPowerOf2_threads_per_row':tpr,'row_lengths':{'min':min(lengths),'max':max(lengths),'empty_rows':lengths.count(0),'histogram':dict(collections.Counter(lengths))},'blocksizes':[32,64,128,256,512,1024],'methods':['dense','csr','vector_csr'],'original_full_CLI':'nnz16777216,n10240,R100 unmeasured','scope':'NamedN512fiveoriginalgenerator densities3/7/15/31/63; sameRAND/drand48sourcefixtures, allrowgroupscompleteavoidnativepartialwarpfullmaskambiguity. ScalarCSR/dense scopeincludesemptyrows wheregenerated; vectorrpr<=1/BS0undefinedcasesunverified.'})
assert off==len(raw);out=array.array('d',expected);(p/'expected_f64.bin').write_bytes(out.tobytes());payload={'cases':desc,'source_fixture':'UnchangedCutils.cpp init_matrix/init_vector/init_csr withsrand48(4096)andinternalrand123 resetpercase; sourcefloatmarkers exact at512²','fixture_sha256':hashlib.sha256(raw).hexdigest(),'input_prefix_count':prefix,'output_values':len(expected),'physical_output_guard_values':16*90,'independent_oracle':'Pythonmath.fsum ofexactFP32products, independentdensevsCSRtraversals checked; FP64referencecomparisonbudget atol1e-5,rtol1e-5; allmodels same. Guards16777215 exact.','native_vector_source_defect':'Everylane writesdifferentpartialsum tosamey[m] aftershuffle; minimalcorrectionlane0singlewriter requiresconditionalnativeflag. Originaldiagnosticretainedbeforecorrection.','timing':'90pathsum, eachR100kernelloopmeanwithblockingdevicewidecompletion, alloc/inputupload/guards/JIT/readback/oracleoutsideROI','precision':'FP32storage/compute, native64bitsize_tindices, Float64losslessartifactencoding/CPUoracle NOTFP64GPUwork'};(p/'fixture_description.json').write_text(json.dumps(payload,indent=2)+'\n');print({'prefix':prefix,'outputs':len(expected),'bytes':len(raw)})
