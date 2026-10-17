"""Independent Float64 stacked reset-after GRU and analytic decay witness."""
from pathlib import Path
import array,math,json,sys
CASES=[(3,3,37,35,3,False),(4,2,5,7,2,True)]
def fixture(S,B,I,H,L,zero):
    M=max(I,H);sizes=[S*B*I,L*3*H*M,L*3*H*H,L*3*H,L*3*H,L*B*H]
    a=[[(17*i+13*idx)%31-15 for i in range(n)]for idx,n in enumerate(sizes,1)]
    for idx,values in enumerate(a,1):a[idx-1]=[x/(128 if idx in (2,3) else 64)for x in values]
    if zero:
        for idx in range(1,5):a[idx]=[0.]*sizes[idx]
    else:
        for l in range(1,L):
            for g in range(3):
                for h in range(H):
                    for k in range(H,M):a[1][((l*3+g)*H+h)*M+k]=55.25
    return a

def reference(a,S,B,I,H,L):
    inp,wi,wh,bi,bh,h0=a;M=max(I,H);hidden=h0.copy();output=[];max_gate=0.;min_gate=1.
    for t in range(S):
        current=inp[t*B*I:(t+1)*B*I];size=I
        for l in range(L):
            next=[]
            for b in range(B):
                for h in range(H):
                    ig=[bi[(l*3+g)*H+h]+math.fsum(current[b*size+k]*wi[((l*3+g)*H+h)*M+k]for k in range(size))for g in range(3)]
                    hg=[bh[(l*3+g)*H+h]+math.fsum(hidden[(l*B+b)*H+k]*wh[((l*3+g)*H+h)*H+k]for k in range(H))for g in range(3)]
                    r=1/(1+math.exp(-ig[0]-hg[0]));z=1/(1+math.exp(-ig[1]-hg[1]));candidate=math.tanh(ig[2]+r*hg[2]);min_gate=min(min_gate,r,z);max_gate=max(max_gate,r,z);old=hidden[(l*B+b)*H+h];ht=(1-z)*candidate+z*old
                    assert min(candidate,old)-1e-15<=ht<=max(candidate,old)+1e-15
                    next.append(ht)
            hidden[l*B*H:(l+1)*B*H]=next;current=next;size=H
        output.extend(current)
    return output+hidden,{'min_gate':min_gate,'max_gate':max_gate,'convex_hidden_blend':True,'final_hidden_bound':max(map(abs,hidden))}

def main(path):
    path=Path(path);r=json.loads(path.read_text());prefix=[];expected=[];quality=[];lengths=[]
    for S,B,I,H,L,zero in CASES:
        a=fixture(S,B,I,H,L,zero)
        for x in a:prefix.extend(x)
        prefix.extend([S,B,I,H,L]);out,q=reference(a,S,B,I,H,L);q['shape']=[S,B,I,H,L];q['zero_weight']=zero
        if zero:
            analytic=[]
            for t in range(1,S+1):analytic.extend(x*2**(-t)for x in a[5][(L-1)*B*H:L*B*H])
            analytic.extend(x*2**(-S)for x in a[5]);assert out==analytic;q['analytic_h_t_equals_half_power_t_h0_exact']=True
        expected.extend(out);lengths.append(len(out));quality.append(q)
    assert len(prefix)==24708 and len(expected)==714;prefixbytes=array.array('f',prefix).tobytes();records=[]
    for model,m in r['models'].items():
        assert len(m['trials'])==5
        for tr in m['trials']:
            raw=Path(tr['output_artifact']['saved_path']).read_bytes();assert raw[:24708*4]==prefixbytes;a=array.array('f');a.frombytes(raw);assert len(a)==25422;got=a[24708:];err=max(abs(x-y)for x,y in zip(got,expected));assert err<=1e-4
            assert list(got[lengths[0]:])==expected[lengths[0]:]
            records.append({'model':model,'trial':tr['trial'],'all_sequence_and_final_hidden_values':714,'exact_input_prefix_bytes':98832,'max_abs_float64_recurrence_error':err,'analytic_zero_case_exact':True})
    report={'result_path':str(path.resolve()),'all_model_trials':records,'recurrence_witnesses':quality,'oracle':'IndependentFloat64scalarGRUequationswithmath.fsumandconvexblendform;GPUwarpstrided32lanedotreductions. Analyticzeroweight h_t=2^-t*h0 exact.','scope':'Twoodd/evenseqlength/nonsquare/stacked/tailhiddenfixtures;alltimeoutputs/alllayersfinalhidden,notinternalgateartifactsorbackpropagation. Paddingpoisonignoredexplicitly.','mapping':'Nativewarp32/fullmask/fourwarpsperblockalignmentP1fromoldgenuineone-threadperhiddenJulia;KA/JACCCUDAshufflemappinghybrid.'}
    out=path.parent/'scientific_recurrence_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main(sys.argv[1])
