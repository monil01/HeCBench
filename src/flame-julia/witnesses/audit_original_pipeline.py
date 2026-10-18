"""CPU audit of preserved unchanged-native integer-storage quantization evidence."""
from pathlib import Path
import struct,json,bisect,hashlib,math,sys
folder=Path(sys.argv[1]);raw=(folder/'params.bin').read_bytes();assert len(raw)==1928
colors=struct.unpack_from('<20f',raw,800);mapping=struct.unpack_from('<20i',raw,1848);n=16384;functions=[]
for i in range(n):
 idx=i&~31;start,end=0,20
 for _ in range(6):
  f=(start+end)//2
  if mapping[f]<=idx:start=f+1
  else:end=f
 assert f==bisect.bisect_right(mapping,idx) and f in [0,1,2];functions.append(f)
init=struct.unpack('<%sh'%n,(folder/'initialized_short_colors.bin').read_bytes());final=struct.unpack('<%sh'%n,(folder/'final_short_colors.bin').read_bytes());assert set(init)==set(final)=={0}
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def f16(x):return struct.unpack('<e',struct.pack('<e',x))[0]
intended=[f16(f32(f32(.5+colors[f])/2)) for f in functions]
assert all(x>0 for x in intended);assert all(int(x)==s for x,s in zip(intended,init))
vertices=folder/'vertices.bin';assert vertices.stat().st_size==n*64*12;nonfinite=0;max_color_error=0
for vi,(x,y,c) in enumerate(struct.iter_unpack('<3f',vertices.read_bytes())):
 nonfinite+=not all(map(math.isfinite,[x,y,c]));expected=f32(colors[functions[vi%n]]/2);max_color_error=max(max_color_error,abs(c-expected))
assert max_color_error==0
rows=[dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest(),bytes=x.stat().st_size) for x in folder.iterdir() if x.is_file()]
print(json.dumps(dict(source_scope='OriginalunchangedN16384oneframe initialize+15iterate+64pointkernels, selectedlegalfrozenpermutation/randomusedprefix inputcontract; notdefaultCLIrandomstream/full1000frames',original_initialized_nonzero_short_colors=sum(x!=0 for x in init),original_final_nonzero_short_colors=sum(x!=0 for x in final),intended_half_initialized_colors_nonzero=n,initialized_color_max_bias_vs_true_half=max(abs(s-t) for s,t in zip(init,intended)),initialized_color_min_bias_vs_true_half=min(abs(s-t) for s,t in zip(init,intended)),vertex_color_max_error_vs_actual_integer_storage_source_model=max_color_error,nonfinite_vertices=nonfinite,actual_function_counts={str(f):functions.count(f) for f in set(functions)},proof='GPUmatchesactualnumericshortconversionbehavior; disagreeswithintendedtruehalfAPIsemantics, notclaimnativeundefinedmemory/GPUfailure. Nooutputguardsinventedfororiginaldiagnostic.',artifacts=rows),indent=2))
