from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent
specs=[(34,64,8,32,1),(514,128,4,64,1),(2050,256,3,128,1),(34,64,8,64,0),(514,128,4,64,0),(2050,256,3,128,0),(34,64,8,32,1)]
prefix=[len(specs)];gold=[];cases=[];HALF_GUARD=-123.;WORD_GUARD=1515870810
for ci,(M,K,B,G,biased) in enumerate(specs):
 tile=32 if biased else 64;groups=K//G;assert K%tile==0 and G%tile==0 and M%2==0
 xu=[(2,1,-1,2,-2,1,1,-1)[(k+ci)%8] for k in range(K)]
 if ci==6:xu=[0]*K
 alpha=[1 if (g*13+b*7+m*3)%5<3 else -1 for g in range(groups) for b in range(B) for m in range(M)];bias=[1 if (g+m)%3 else -1 for g in range(groups) for m in range(M)] if biased else []
 W=[]
 for kt in range(K//32):
  for b in range(B):
   for m in range(M):
    x=((kt+1)*0x9e3779b9^(b+3)*0x85ebca6b^(m+7)*0xc2b2ae35)&0xffffffff;x^=x>>16;x=(x*0x7feb352d)&0xffffffff;x^=x>>15
    if m<6:x=[0,0xffffffff,0xaaaaaaaa,0x55555555,0x01234567,0x80000001][m]
    W.append(x)
 out=[]
 for m in range(M):
  total=0
  for k in range(K):
   g=k//G;coeff=bias[g*M+m] if biased else 0
   for b in range(B):coeff+=alpha[g*B*M+b*M+m]*(1 if (W[(k//32)*B*M+b*M+m]>>(k%32))&1 else -1)
   total+=xu[k]*coeff
  assert abs(total)<=2048;out.append(total/512)
 bound=K*max(map(abs,xu),default=0)*(B+biased);assert bound<=2048
 # AllLUT/partial/scale/atomicintermediates are exacthalf: quarterintegerLUT,
 # alpha/q_bias1/128, outputs integerquantum1/512 withabsolute integerbound2048.
 arrays=[W,[v/128 for v in alpha],[v/128 for v in bias],[v/4 for v in xu]]
 prefix +=[M,K,B,G,biased,tile]
 for a in arrays:prefix+=a
 gold +=[HALF_GUARD]*8+out+[HALF_GUARD]*8
 gold +=[WORD_GUARD]*8+W+[WORD_GUARD]*8
 for a in arrays[1:]:gold +=[HALF_GUARD]*8+a+[HALF_GUARD]*8
 cases.append(dict(case=ci,M=M,K=K,bits=B,group_size=G,groups=groups,bias=bool(biased),K_tile=tile,M_tile=2048,threads=256,grid=[(M+2047)//2048,K//tile],array_sizes=list(map(len,arrays)),zero_input=ci==6,output_min=min(out),output_max=max(out),proof_absolute_output_quantum_bound=bound,quantum=1/512,all_packed_bit_positions_have_nonzero_inputs=ci!=6))
for name,a in [('fixture.bin',prefix),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(a),*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),oracle='Independentdensepacked-bitdequantization andintegerweightedmatvec in1/512units, noLUTsourcecopy. Quarterinputintegerxu∈{-2,-1,1,2}, halfalpha/qbias±1/128, all8/4/3planesnonzeroscale.',exact_half_proof='BoundK*maxabs(xquarter)*(bits+hasbias)<=2048; allatomicpartialsumorders lie[-2048,2048]quantum1/512, representableFloat16includingboundary4. EveryLUTsumquarterinteger andscaledpartialon1/512grid; no hiddenFP32accumulation permitted.',atol=0,rtol=0,source_support='Mpositiveeven,Kmultiples32bias/64unbiased, groupsaligntiles. NooddM/partialKtileknownOOBGPUrun; selectedrawpackedbitplanesall32positionsactiveexceptzero-Xcontrol.',timing='Outputclear+actualLUTkernel+completion included; canary.5payloadresetoutsideROIprovesclearneeded, guard-123exacthalf'),indent=2)+'\n')
print(json.dumps(cases))
