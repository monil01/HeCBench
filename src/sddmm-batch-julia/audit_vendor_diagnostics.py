from pathlib import Path
import struct,json,hashlib
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;q=p/'reference_history/diverse_row_offsets';sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();pre=list(struct.unpack('<78637d',(q/'fixture.bin').read_bytes()));gold=list(struct.unpack('<81442d',(q/'oracle.bin').read_bytes()));desc=json.loads((q/'fixture_description.json').read_text());alternative=[];pos=1
for case in desc['cases']:
 b,m,k,n,nz,rep=map(int,pre[pos:pos+6]);pos+=6;A=pre[pos:pos+b*m*k];pos+=len(A);B=pre[pos:pos+b*k*n];pos+=len(B);off=list(map(int,pre[pos:pos+b*(m+1)]));pos+=len(off);col=list(map(int,pre[pos:pos+b*nz]));pos+=len(col)+b*nz
 predicted=[]
 for batch in range(b):
  for row in range(m):
   for j in range(off[row],off[row+1]):predicted.append(sum(A[batch*m*k+row*k+h]*B[batch*k*n+h*n+col[batch*nz+j]] for h in range(k)))
 alternative.append(predicted)
rows=[]
for run,model,version in [('20261018T114142648182Z_sddmm-batch','cuda_jl',12001),('20261018T114308372762Z_sddmm-batch','cuda',12001),('20261018T114507630120Z_sddmm-batch','cuda_jl',12510)]:
 bundle=r/'.porting-state/portable_julia/runs'/run;d=json.loads((bundle/'result.json').read_text());cfg=json.loads((bundle/'config.json').read_text());assert len(d['models'][model]['trials'])==5
 for t in d['models'][model]['trials']:
  f=Path(t['output_artifact']['saved_path']);data=f.read_bytes();assert len(data)==160079*8;actual=list(struct.unpack('<160079d',data));assert actual[:78637]==pre;actual=actual[78637:];off=0;cases=[]
  for ci,c in enumerate(desc['cases']):
   out=actual[off:off+c['C_values']];expected=gold[off:off+c['C_values']];mismatch=sum(x!=y for x,y in zip(out,expected));maxabs=max(abs(x-y) for x,y in zip(out,expected));assert out==alternative[ci];assert actual[off+c['C_values']:off+c['oracle_values']]==gold[off+c['C_values']:off+c['oracle_values']];off+=c['oracle_values'];cases.append(dict(case=ci+1,actual_outside_expected_contract_mismatches=mismatch,max_absolute_error=maxabs,exactly_matches_first_batch_row_offsets_with_per_batch_columns_and_dense_inputs=True))
  assert not t['correct'] and sum(c['actual_outside_expected_contract_mismatches'] for c in cases)>0
  rows.append(dict(run_id=run,model=model,vendor_api_version=version,trial=t['trial'],result_sha256=sha(bundle/'result.json'),config_sha256=sha(bundle/'config.json'),artifact=dict(path=str(f.relative_to(r)),sha256=sha(f),bytes=len(data)),all_actual_inputs_initialC_and_guards_exact=True,cases=cases))
out=dict(benchmark='sddmm-batch',diagnostic_only=True,all15_failed_captures_preserved=True,framework_unsupported_claim=False,portable_translation_attempts=0,scope='DifferentCSRrowoffsetdistributions acrossbatches notoriginalinitializerdomain. SourcepatternCPUwitness actualstructurechanges0/valueschanges3 across4calls. Retainstrictoriginalrationaloracle; noaccuracyrelaxation.',mathematical_target='Eachbatch ownrowoffsets/columns/denseA/B sampledproduct.',observed_behavior='Everyoutput of15failedcaptures acrossnative+Julia/vendor12001+12510 exactlymatches batch0rowoffsets plus correctperbatchcolumns/A/B. Entireinput/initialC/guards unchanged. APIreturnsSUCCESS, this doesnotestablishwhyvendorignoresoffsetstride.',documentation=dict(url='https://docs.nvidia.com/cuda/archive/12.0.0/cusparse/index.html#cusparsesddmm',section='14.6.13',paraphrase='SDDMMsupportsbatched CSR usingCsrSetStridedBatch andDnMatSetStridedBatch, noexplicitsharedrowoffsetrestriction found inreviewedsection.',is_inference='Observedbehavior not universalAPI/hardware/frameworklimit; onlyactualvendorversions/device tested.'),fixtures=[dict(path=str(f.relative_to(r)),sha256=sha(f)) for f in [q/'fixture.bin',q/'oracle.bin',q/'generate_study_fixture.py',q/'fixture_description.json']],rows=rows)
(p/'vendor_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n');print('15captures proven first-batch-offset hypothesis, original oracle failures retained')
