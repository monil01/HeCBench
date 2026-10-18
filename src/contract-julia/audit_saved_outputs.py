"""Read-only saved output/byte-input/source audit for both contraction precisions."""
from pathlib import Path
import hashlib,json,struct,sys,math
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
paths=[Path(x) for x in sys.argv[1:]]
if not paths:paths=[R/x['result'] for x in json.loads((P/'configuration_results.json').read_text())]
tensor=(P/'study_tensor_f64.bin').read_bytes();adj=(P/'study_adj_f64.bin').read_bytes();oracle=(P/'oracle_f64.bin').read_bytes();records=[];sources=[]
for rp in paths:
 r=json.loads(rp.read_text());bits=r['input_contract']['precision_bits'];expected=tensor+adj+struct.pack('<4d',9,18,18,bits)+oracle+struct.pack('<64d',*([-8192]*64));assert len(expected)==39515*8
 assert r['reference_agreement'] and all(r['variant_verified'].values()) and r['matched_performance_valid']
 for m,v in r['models'].items():
  assert len(v['trials'])==5 and v['provenance_valid']
  for t in v['trials']:
   a=t['output_artifact'];f=Path(a['saved_path']);raw=f.read_bytes();assert raw==expected and hashlib.sha256(raw).hexdigest()==a['sha256']
   vals=struct.unpack('<39515d',raw);assert all(map(math.isfinite,vals));records.append(dict(precision_bits=bits,model=m,trial=t['trial'],artifact=str(f.relative_to(R)),sha256=a['sha256'],exact_input_prefix_bytes=13207*8,exact_output_values=26244,exact_guard_values=64,independent_oracle_max_abs_error=0))
  for filename,h in v['adapter_file_hashes'].items():
   f=Path(filename);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;sources.append(dict(model=m,precision_bits=bits,path=filename,sha256=h))
result=dict(benchmark='contract',results=[str(x.relative_to(R)) for x in paths],records=records,source_bindings=sources,all40_full_artifacts_exact=len(records)==40,all_sources_fixtures_automatically_bound=True,oracle='Independent NumPy einsum18formulae,exactdyadicproducts/sums; notdevicebranchlooptranscription',scientific_scope='All18sourcecontractionoperators,bothprecisions,namedN9C18;notoriginalN64orapplicationquality')
(P/'saved_output_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(artifacts=len(records),source_bindings=len(sources))))
