"""Independent global pair-rotation oracle: no native tile algorithm imported."""
from pathlib import Path
import math,struct,json,hashlib
P=Path(__file__).resolve().parent

def layers(real,imag,w,h,a,b,steps):
    current=[complex(r,i) for r,i in zip(real,imag)];previous=[0j]*len(current)
    for step in range(steps):
        previous=current[:]
        # Four disjoint nearest-neighbor matchings, then their reverse.
        for axis,direction in [(1,1),(0,1),(1,-1),(0,-1),(0,-1),(1,-1),(0,1),(1,1)]:
            for y in range(h):
                for x in range(w):
                    if (x+y)%2:continue
                    px=x+(direction if axis==0 else 0);py=y+(direction if axis==1 else 0)
                    if not(0<=px<w and 0<=py<h):continue
                    i=y*w+x;j=py*w+px;u=current[i];v=current[j]
                    current[i]=a*u+complex(0,b)*v;current[j]=a*v+complex(0,b)*u
    return current,previous

def main():
    metadata=[];allinput=[];allexpected=[]
    for T in ['float32','float64']:
        cast=lambda x:struct.unpack('<f',struct.pack('<f',x))[0] if T=='float32' else x
        a=cast(math.cos(.02));b=cast(math.sin(.02))
        for name,w,h,steps in [('source_gaussian',64,96,3),('signed_dyadic',70,98,4),('boundary_impulses',34,50,1)]:
            re=[];im=[]
            for y in range(h):
                for x in range(w):
                    if name=='source_gaussian':
                        amp=math.exp(-((x+1-180.)**2+(y+1-300.)**2)/(2*64.**2));phase=.4*(x+y+2-480.)
                        r=amp*math.cos(phase);v=amp*math.sin(phase)
                    elif name=='signed_dyadic':r=((17*x+13*y)%127-63)/128;v=((7*x-19*y)%131-65)/128
                    else:r=(1. if(x,y)==(0,0)else -.5 if(x,y)==(w-1,h-1)else 0.);v=(.75 if(x,y)==(w//2,h//2)else 0.)
                    re.append(cast(r));im.append(cast(v))
            out,prev=layers(re,im,w,h,a,b,steps);norm0=math.fsum(r*r+i*i for r,i in zip(re,im));norm=math.fsum(abs(z)**2 for z in out)
            # Exact-real pair matrix has scale a^2+b^2; rounded FP32 coefficients slightly nonunitary.
            inverse,_=layers([z.real for z in out],[z.imag for z in out],w,h,a,-b,steps)
            metadata.append({'dtype':T,'name':name,'width':w,'height':h,'steps':steps,'a':a,'b':b,'input_offset':len(allinput),'output_offset':len(allexpected),'norm_initial':norm0,'oracle_norm_final':norm,'roundtrip_max_abs':max(abs(z-complex(r,i))for z,r,i in zip(inverse,re,im)),'pair_scale_a2_b2':a*a+b*b,'oracle_output_atol':1e-5 if T=='float32'else 1e-11})
            allinput +=[a,b]+re+im
            # Capture active final then previous. Before first step other buffer is zero but is fully written.
            allexpected +=[z.real for z in out]+[z.imag for z in out]+[z.real for z in prev]+[z.imag for z in prev]
    for name,values in [('fixture_f64.bin',allinput),('expected_f64.bin',allexpected)]:
        (P/name).write_bytes(struct.pack('<'+'d'*len(values),*values))
    d={'cases':metadata,'input_count':len(allinput),'output_count':len(allexpected),'fixture_sha256':hashlib.sha256((P/'fixture_f64.bin').read_bytes()).hexdigest(),'expected_sha256':hashlib.sha256((P/'expected_f64.bin').read_bytes()).hexdigest(),'scope':'Six named FP32/FP64cases with3/4/1physicalsplitsteps, fullsource16x16threadtiles,BY96/32,halo3x4. Original1024x1024x100 unmeasured.','oracle':'Independent global complex rotations over disjoint nearest-neighbor edge matchings12344321; Float64fromexactstoredinputs/typedcoefficients, not tiledsource/float-truncatedsourceCPUchecker','coefficient_caveat':'TypedFP32cos/sincoefficients neednot satisfyexacta2+b2=1; report actualnormdrift, donotdemand impossibleexactunitarity. No timecontinuumHamiltonian/physicalaccuracyclaim.'}
    (P/'fixture_provenance.json').write_text(json.dumps(d,indent=2)+'\n');print({'input':len(allinput),'output':len(allexpected)})
if __name__=='__main__':main()
