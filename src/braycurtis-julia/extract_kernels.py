"""Extract exact unchanged device-function text, excluding unavailable ATen host wrappers."""
from pathlib import Path
import re,hashlib,json
P=Path(__file__).resolve().parent;R=P.parents[1];source=R/'src/braycurtis-cuda/torchpairwise/csrc/ops/cuda/braycurtis_kernel.cu';s=source.read_text();names=['_braycurtis_forward_kernel_impl','_braycurtis_forward_components_kernel_impl','_braycurtis_backward_x1_kernel_impl','_braycurtis_backward_x2_kernel_impl'];parts=[];spans=[]
for name in names:
 pos=s.index('void '+name+'(');start=s.rfind('template<typename scalar_t, typename index_t>',0,pos);brace=s.index('{',pos);depth=1;end=brace+1
 while depth:
  depth+=(s[end]=='{')-(s[end]=='}');end+=1
 text=s[start:end];parts.append(text);spans.append(dict(name=name,start_byte=start,end_byte=end,sha256=hashlib.sha256(text.encode()).hexdigest(),arithmetic_and_body_unchanged=True))
out='// Exact source device functions; see extraction_manifest.json for spans and hashes.\n'+ '\n\n'.join(parts)+'\n';(P/'source_kernels.inc').write_text(out)
(P/'extraction_manifest.json').write_text(json.dumps(dict(original_source=str(source.relative_to(R)),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),functions=spans,extracted_file_sha256=hashlib.sha256(out.encode()).hexdigest(),shim_scope='Row-major packed accessor size/stride/indexing + source grid-stride macro. No tensor dispatcher/autograd/ATen host execution; source Float32/64 arithmetic unchanged; Float16 unmeasured.'),indent=2)+'\n');print('BRAY_EXACT4_EXTRACTED')
