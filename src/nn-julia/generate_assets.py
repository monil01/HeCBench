"""Deterministic planar-coordinate fixtures + independent hypot/global-sort oracle."""
from pathlib import Path
import struct,math,json,hashlib,sys
P=Path(sys.argv[1])if len(sys.argv)>1 else Path(__file__).resolve().parent;P.mkdir(parents=True,exist_ok=True)
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
Ns=[1,17,65,4099,131072];qs=[(30,90),(-10,-20),(0,0),(30,90),(30,90)];R=10000;guard=f32(91.25)
cases=[];data=[];prefix=[];outputs=[];masks=[];off=0
for ci,(n,(lat,lng))in enumerate(zip(Ns,qs)):
 lat,lng=f32(lat),f32(lng);pts=[(f32(-60+(i%97)/2),f32(-170+((i*37)%701)/2))for i in range(n)]
 ids=[0]if n==1 else[n-1,n//2,0,3,5]
 for rank,i in enumerate(ids):pts[i]=(f32(lat+rank*.25),lng)
 packed=[(guard,guard)]*8+pts+[(guard,guard)]*8
 vals=[v for p in packed for v in p];data+=vals
 row=dict(n=n,k=min(5,n),repeats=R,lat=lat,lng=lng,input_offset_floats=off,input_float_count=len(vals));cases.append(row);off+=len(vals)
 header=[n,row['k'],R,bits(lat),bits(lng),bits(lat),bits(lng),bits(f32(n))]
 rb=[bits(x)for x in vals];prefix+=header+rb+rb
 distances=[math.hypot(float(lat)-x,float(lng)-y)for x,y in pts]
 selected=sorted(range(n),key=lambda i:(distances[i],i))[:row['k']]
 assert len({distances[i]for i in selected})==row['k']
 permutation=list(range(n))
 for i,want in enumerate(selected):j=permutation.index(want);permutation[i],permutation[j]=permutation[j],permutation[i]
 out=distances+[distances[i]for i in permutation]+permutation+[bits(guard)]*32
 mask=[int(v==0)for v in distances]+[int(distances[i]==0)for i in permutation]+[1]*(n+32)
 outputs+=out;masks+=mask
prefix_count=len(prefix);expected=prefix+outputs;mask=[1]*prefix_count+masks
(P/'input_f32.bin').write_bytes(struct.pack('<'+'f'*len(data),*data))
(P/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(expected),*expected))
(P/'oracle_exact_u8.bin').write_bytes(bytes(mask))
h='struct NNCase {int n,k,repeats,offset;float lat,lng;};\nstatic const NNCase cases[] = {\n'+''.join('{%d,%d,%d,%d,%.9gf,%.9gf},\n'%(c['n'],c['k'],c['repeats'],c['input_offset_floats'],c['lat']+0.0,c['lng']+0.0)for c in cases)+'};\n'
# C++ requires a decimal/exponent on floating literals.
h=h.replace('30f','30.0f').replace('90f','90.0f').replace('-10f','-10.0f').replace('-20f','-20.0f').replace(',0f',',0.0f')
h+='static constexpr size_t INPUT_FLOATS=%d,PREFIX_COUNT=%d,OUTPUT_COUNT=%d;\n'%(len(data),prefix_count,len(expected));(P/'study_cases.h').write_text(h)
j='const CASES = [\n'+''.join('(n=Int32(%d),k=%d,repeats=%d,offset=%d,lat=Float32(%r),lng=Float32(%r)),\n'%(c['n'],c['k'],c['repeats'],c['input_offset_floats'],c['lat'],c['lng'])for c in cases)+']\nconst INPUT_FLOATS=%d\nconst PREFIX_COUNT=%d\nconst OUTPUT_COUNT=%d\n'%(len(data),prefix_count,len(expected));(P/'study_cases.jl').write_text(j)
files={f:{'bytes':(P/f).stat().st_size,'sha256':hashlib.sha256((P/f).read_bytes()).hexdigest()}for f in ['input_f32.bin','oracle_f64.bin','oracle_exact_u8.bin','study_cases.h','study_cases.jl']}
(P/'study_plan.json').write_text(json.dumps({'cases':cases,'input_prefix_count':prefix_count,'artifact_count':len(expected),'files':files,'generator':'Pythonstdlibstruct exactFloat32; independentmath.hypot Float64 andglobalsort unique top5, reconstruct fullpermutation bydesiredIDs notsourceFloat32minscan','oracle_acceptance':'Exactinputprefix/rawguards/IDs/zerodistances; otherdistances abs<=1e-5+8epsFloat32*abs(reference); finiteallvalues. Predeclareddualsamebudget.','reference_repair':'OwnedminimalCPUrecordvalueswap; sourceGPUbodyunchanged.','source_cli_scope':'Originalexternaldatabase unavailable; namedtypedcoordinates ratherthanfulloriginaldataset/fileparser throughput. Originalsynthetic48charCLI/sourcebug actualwitness separate.','metric':'PlanarEuclidean lat/lng coordinateunits, notgreatcircle distance.','storage_precision':'Float32AoS inputanddistance; Float64 artifact encoding, UInt32 inputbits losslessnumericF64; Float64 independenthypot outsideROI.'},indent=2)+'\n')
print('NN_ASSETS',prefix_count,len(expected),files)
