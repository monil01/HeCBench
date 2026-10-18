
from pathlib import Path
import numpy as np,json,hashlib
p=Path(__file__).resolve().parent
metadata=[]
for case,(m,k,b)in enumerate([(12288,4096,4),(96,512,4)],1):
 values=np.fromfile(p/f"study_case{case}_u16.bin",dtype="<u2");nc=m*k//8;ni=b*k
 codes=values[:nc].reshape(m,k//8);inputs=values[nc:nc+ni].view("<f2").reshape(b,k);books=values[nc+ni:].view("<f2").reshape(3,65536,8)
 output=np.empty((b,m),dtype="<f2");dense=np.empty((b,m),dtype="<f8")
 for first in range(0,m,256):
  end=min(first+256,m);parts=np.arange(first,end)//(m//3);cb=codes[first:end];rr=len(parts)
  # Unquantized mathematical dot independently reconstructs every codebook block and uses FP64 summation.
  decoded=books[parts[:,None],cb,:].astype("f8").reshape(rr,k)
  for batch in range(b):
   dense[batch,first:end]=decoded@inputs[batch].astype("f8")
   lane=np.zeros((rr,32),dtype="f4")
   # Exact operational oracle preserves packed-half2 four fused updates per codebook-entry then FP32 warp tree.
   for tile in range((k//8+255)//256):
    for enc in range(8):
     active=np.arange(32);packed=tile*32+active;valid=packed<k//64
     code_index=packed[valid]*8+enc
     decoded_block=books[parts[:,None],cb[:,code_index],:]
     x=inputs[batch,(tile*256+active[valid]*8+enc)[:,None]*8+np.arange(8)]
     even=np.zeros((rr,int(np.sum(valid))),dtype="f2");odd=even.copy()
     for j in range(4):
      # Binary16 operands under fixture ranges have exact product+sum in Float64 before one RNEhalfcast.
      even=(decoded_block[:,:,2*j].astype("f8")*x[None,:,2*j].astype("f8")+even.astype("f8")).astype("f2")
      odd=(decoded_block[:,:,2*j+1].astype("f8")*x[None,:,2*j+1].astype("f8")+odd.astype("f8")).astype("f2")
     lane[:,valid]=lane[:,valid]+(even.astype("f4")+odd.astype("f4"))
     # C source: res += (Float32(even)+Float32(odd)); enforce that inner addition first, not left-associative res+even+odd.
     # Assignment above is replaced below by explicit source-grouped accumulation in generated source.
   for off in (16,8,4,2,1):lane[:,:32-off]=lane[:,:32-off]+lane[:,off:]
   output[batch,first:end]=lane[:,0].astype("f2")
 (p/f"oracle_case{case}_u16.bin").write_bytes(output.view("<u2").tobytes())
 (p/f"dense_case{case}_f64.bin").write_bytes(dense.tobytes())
 diff=np.abs(output.astype("f8")-dense)
 metadata.append({"case":case,"shape":[b,m],"output_dtype":"IEEEbinary16bits","operational_oracle":"CPUbookdecode/individualexacthalfFMArounding/FP32laneaccumulation/sourcewarpdown16,8,4,2,1/finalhalfcast","independent_mathematical_operator":"Fullyreconstructed FP64densematrixvectorproduct, differentprecision/reductionorder; notprimarybitcorrectness","native_precision_vs_dense_maxabs":float(diff.max()),"native_precision_vs_dense_RMSE":float(np.sqrt(np.mean(diff*diff))),"operational_output_sha256":hashlib.sha256((p/f"oracle_case{case}_u16.bin").read_bytes()).hexdigest(),"dense_output_sha256":hashlib.sha256((p/f"dense_case{case}_f64.bin").read_bytes()).hexdigest(),"finite":bool(np.isfinite(output).all())})
(p/"oracle_provenance.json").write_text(json.dumps({"NumPy":np.__version__,"cases":metadata,"CPU_generated":True,"half_fma_exactness_scope":"Finitefixturehalfvalues bounded0..1 orsigneddyadic1/256and1/512; Float64representsbinary16product+halfaccsumexactlyinrangebeforeRNhalf. Notuniversalhalffmaemulator."},indent=2)+"\n")
print(metadata)
