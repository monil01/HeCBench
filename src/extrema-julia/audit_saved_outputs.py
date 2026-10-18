"""CPU-only raw typed input/Bool mask/guard/coordinate oracle and five-trial audit."""
from pathlib import Path
import json,struct,hashlib,sys,math
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();meta=json.loads((p/'fixture_description.json').read_text());fix=struct.unpack('<3260Q',raw);gold=struct.unpack('<120712Q',(p/'oracle.bin').read_bytes());arrays=[];pos=0
for c in meta['arrays']:
 ty,dim,nx,ny=fix[pos:pos+4];pos+=4;n=nx*ny;bits=fix[pos:pos+n];pos+=n;fmt=['i','q','f','d'][ty-1];ufmt='I' if ty in [1,3] else 'Q';values=[struct.unpack('<'+fmt,struct.pack('<'+ufmt,x))[0] for x in bits];arrays.append((bits,values))
assert pos==3260
# Recompute every mask independently from coordinates/mathmodulo, not source clipping helper.
expected=[];payload_count=0
for path in meta['paths']:
 bits,v=arrays[path['array']];nx,ny,n,order,axis,clip=[path[k] for k in ['nx','ny','n','order','axis','clip']];dimension=path['dimension'];ty=path['type'];mask=[]
 def coord(x,N):return min(max(x,0),N-1) if clip else x%N
 for i,x in enumerate(v):
  row,col=divmod(i,nx);pairs=[]
  for o in range(1,order+1):
   if dimension==1:a,b=coord(i+o,n),coord(i-o,n)
   elif axis==0:a,b=coord(row+o,ny)*nx+col,coord(row-o,ny)*nx+col
   else:a,b=row*nx+coord(col+o,nx),row*nx+coord(col-o,nx)
   pairs.append(x>v[a] and x>=v[b])
  mask.append(int(all(pairs)))
 assert sum(mask)==path['true_count'];assert all(mask[j]==0 for j,x in enumerate(v) if isinstance(x,float) and math.isnan(x))
 if nx==1 and axis==1:assert not any(mask)
 if path['array']%3==0 and order==1:assert mask[0:2]==[0,1] # lastplateaumember selected
 gb=0xA5A5A5A5 if ty in [1,3] else 0xA5A5A5A5A5A5A5A5
 expected += [165]*8+mask+[165]*8+[gb]*8+list(bits)+[gb]*8;payload_count+=n
assert tuple(expected)==gold and payload_count==57988;rows=[];first=None
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text());assert res['reference_agreement']
 for model,d in res['models'].items():
  assert len(d['trials'])==5 and all(t['correct'] for t in d['trials']);assert abs(d['mean_seconds']-sum(t['time_seconds'] for t in d['trials'][1:])/4)<1e-12
  for t in d['trials']:
   f=Path(t['output_artifact']['saved_path']);data=f.read_bytes();first=first or data;assert data==first and data[:len(raw)]==raw;out=struct.unpack('<120712Q',data[len(raw):]);assert out==gold
   assert t['artifact_word_mismatches']==0 and t['max_abs_error'] is None and t['max_rel_error'] is None
   assert t['path_true_counts']==[x['true_count'] for x in meta['paths']];assert abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   for name,h in d.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,(model,name)
   rows.append(dict(model=model,trial=t['trial'],artifact=str(f),sha256=hashlib.sha256(data).hexdigest(),raw_input_prefix_bytes_exact=True,all_mask_bits_and_typedinputecho_exact=True,all4736physicalguards_exact=True,independent_coordinate_oracle_mask_mismatches=0,uncomputed_float_error_fields_null=True,all_declared_dependencies_current=True))
print(json.dumps(dict(artifact_values=123972,input_prefix_values=3260,input_prefix_bytes=len(raw),actual_mask_bits_pertrial=57988,physical_guards_pertrial=4736,parameter_paths=148,actual_storage_types=['Int32','Int64','Float32','Float64','nativeBool_byte_JuliaUInt8_mask'],input_artifact_encoding='RawUInt64bits, 32bittypestoragezeroextended, noFloat64widenInt64orNaNcanonicalization',known_properties=['asymmetricplateau_lastmember','singletonaxis_strictself_false','NaNvalue_orderedcomparison_false','Int64extremecomparison_noarithmeticoverflow','signedzero_inputbitsretained','clippedandboundedwrapcoordinates'],rows=rows),indent=2))
