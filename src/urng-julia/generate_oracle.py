"""Exact Schrage/LCG shuffle with lane-private tables and staged FP32 rounding."""
from pathlib import Path
import struct,json,hashlib,collections
P=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
M=2147483647;A=16807;Q=127773;IR=2836;NTAB=16;NDIV=134217728

def step(s):
    # C truncation toward zero, including the initial negative seed.
    k=abs(s)//Q*(-1 if s<0 else 1);s=A*(s-k*Q)-IR*k
    return s+M if s<0 else s

def rng(seed):
    state=seed;shuffle=[None]*16
    for j in range(16,-1,-1):
        state=step(state)
        if j<16:shuffle[j]=state
    iy=shuffle[0];state=step(state);j=iy//NDIV;iy=shuffle[j]
    return f32(f32(iy)*f32(1/2**31)),shuffle,j

def main():
    raw=(P/'fixture_rgba.bin').read_bytes();assert len(raw)==(16384+1031)*4;expected=bytearray();cases=[]
    for name,offset,n in [('source_bitmap',0,16384),('all_grays_and_tail',16384,1031)]:
        seeds=[];noises=[];output=bytearray()
        for k in range(n):
            p=raw[4*(offset+k):4*(offset+k+1)];seed=-(sum(p)//4);u,table,j=rng(seed);dev=f32(f32(u-f32(.55))*f32(25));seeds.append(seed);noises.append(dev)
            for x in p:output.append(int(max(0.,min(255.,f32(f32(x)+dev)))))
        mean=sum(int(y)-int(x) for x,y in zip(raw[4*offset:4*(offset+n)],output))/(n*4*25)
        expected+=output+bytes([165])*(17*4)
        cases.append({'name':name,'pixels':n,'source_mean_check':mean,'unique_seeds':len(set(seeds)),'unique_deviations':len(set(noises)),'deviation_min':min(noises),'deviation_max':max(noises),'zero_seed_pixels':seeds.count(0)})
    # Concrete overlapping writes: lane0,j1 andlane16,j0 both index16, unequal logical values.
    _,t0,_=rng(-100);_,t16,_=rng(-101)
    collision={'lane0_seed':-100,'lane16_seed':-101,'source_index_lane0_j1':16,'source_index_lane16_j0':16,'lane0_intended_value_j1':t0[1],'lane16_intended_value_j0':t16[0],'corrected_indices':[256,16],'observed_GPU_race_failure':None,'proof':'OriginalNTAB*j+tid has16stridewith256lanes; differentprivate-tableelementswrite sameaddress. GROUP_SIZE*j+tid givesdisjointownership. No blockbarrier isneededforgenuineprivatecolumns.'};assert t0[1]!=t16[0]
    (P/'expected_rgba_guards.bin').write_bytes(expected)
    meta={'cases':cases,'input_count':len(raw),'output_count':len(expected),'NTAB':16,'FACTOR':25,'GROUP_SIZE':256,'guard_pixels_per_case':17,'original_image':'src/urng-sycl/URNG_Input.bmp','original_image_sha256':hashlib.sha256((P.parent/'urng-sycl/URNG_Input.bmp').read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(raw).hexdigest(),'expected_sha256':hashlib.sha256(expected).hexdigest(),'collision_witness':collision,'oracle':'ExactintegerSchrageupdates andprivate16-entryshuffle, stagedFloat32conversions/deviation/clamp/truncate; alphaalsoaffected, notRGBonly','scientific_scope':'Source deterministicpixel-average-seedednoise, atmost256seeds/deviations, identicalaveragepixelsrepeatsamevalue; zeroseedabsorbing andoffset-.55 meanbias retained. Noindependentidenticallydistributedspatialnoise/statisticalqualityclaim.'}
    (P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['cases']);print(collision)
if __name__=='__main__':main()
