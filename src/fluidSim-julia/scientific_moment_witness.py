"""Independent D2Q9 local-moment and pull-stream full-population witness."""
from pathlib import Path
import array,math,json,sys
W=256;H=32;N=W*H;STEPS=8;OMEGA=array.array('f',[1.2])[0]
DX=[0,1,0,-1,0,1,-1,-1,1];DY=[0,0,1,0,-1,1,1,-1,-1];WEIGHTS=[4/9]+[1/9]*4+[1/36]*4;OPP=[0,3,4,1,2,7,8,5,6]
def eq(r,w,dx,dy,ux,uy):
    u2=ux*ux+uy*uy;eu=dx*ux+dy*uy
    return r*w*(1+3*eu+4.5*eu*eu-1.5*u2)
def encode(f):return [v[0]for v in f]+[v[k]for v in f for k in range(1,5)]+[v[k]for v in f for k in range(5,9)]
def fixture(perturbed):
    f=[];types=[]
    for y in range(H):
        for x in range(W):
            r=1+((17*x+13*y)%9)/64 if perturbed else 1.;ux=1/128 if perturbed else 0.;uy=-1/256 if perturbed else 0.
            f.append([eq(r,w,dx,dy,ux,uy)for w,dx,dy in zip(WEIGHTS,DX,DY)])
            types.append(int(x==0 or x==W-1 or y==0 or y==H-1 or(perturbed and 120<=x<=123 and 12<=y<=19)))
    return f,types

def reference(perturbed):
    f,types=fixture(perturbed);initial=encode(f);prefix=initial+types+[OMEGA]+WEIGHTS+[float(W),float(H)];out=[v.copy()for v in f];mass0=math.fsum(initial);masserr=momerr=0.;bounces=0
    for step in range(STEPS):
        collision=[]
        for pos,old in enumerate(f):
            if types[pos]:
                new=[old[k]for k in OPP]
                if pos%W not in (0,W-1) and pos//W not in (0,H-1):bounces+=1
            else:
                rho=math.fsum(old);mx=math.fsum(a*b for a,b in zip(old,DX));my=math.fsum(a*b for a,b in zip(old,DY));ux=mx/rho;uy=my/rho
                new=[(1-OMEGA)*old[k]+OMEGA*eq(rho,WEIGHTS[k],DX[k],DY[k],ux,uy)for k in range(9)]
                masserr=max(masserr,abs(math.fsum(new)-rho));momerr=max(momerr,abs(math.fsum(a*b for a,b in zip(new,DX))-mx),abs(math.fsum(a*b for a,b in zip(new,DY))-my))
            collision.append(new)
        # Destination-oriented pull independently realizes the native interior-source push policy.
        for y in range(H):
            for x in range(W):
                for k in range(9):
                    sx=x-DX[k];sy=y-DY[k]
                    if 0<sx<W-1 and 0<sy<H-1:out[x+W*y][k]=collision[sx+W*sy][k]
        f,out=out,f
    final=encode(f);scratch=encode(out);stationary=max(abs(a-b)for a,b in zip(final,initial));assert masserr<1e-12 and momerr<1e-12
    if not perturbed:assert stationary<1e-12
    else:assert bounces==32*STEPS and stationary>1e-5
    quality={'perturbed':perturbed,'max_local_fluid_collision_mass_error':masserr,'max_local_fluid_collision_momentum_error':momerr,'interior_obstacle_bounce_cell_steps':bounces,'initial_total_population':mass0,'final_total_population':math.fsum(final),'global_total_change_observation':math.fsum(final)-mass0,'max_change_from_initial':stationary,'minimum_final_population':min(final),'scope':'Localcollisionconservationandactualsource-boundarysemantics;globalclosedwallconservationNOTasserted. Boundarysourcesdonotpropagate;oldunwrittencomponentspersist.'}
    return prefix,final+scratch,quality

def main(path):
    path=Path(path);r=json.loads(path.read_text());prefix=[];outputs=[];quality=[]
    for perturbed in(False,True):
        p,o,q=reference(perturbed);prefix.extend(p);outputs.extend(o);quality.append(q)
    expected=prefix+outputs;assert len(expected)==458776;prefixbytes=array.array('d',prefix).tobytes();records=[]
    for model,m in r['models'].items():
        assert len(m['trials'])==5
        for trial in m['trials']:
            raw=Path(trial['output_artifact']['saved_path']).read_bytes();assert raw[:len(prefixbytes)]==prefixbytes
            a=array.array('d');a.frombytes(raw);assert len(a)==len(expected);err=max(abs(x-y)for x,y in zip(a,expected));assert err<=1e-10 and all(math.isfinite(x)for x in a)
            records.append({'model':model,'trial':trial['trial'],'exact_input_prefix_values':163864,'full_population_outputs':294912,'max_abs_independent_pull_oracle_error':err})
    report={'result_path':str(path.resolve()),'all_model_trials':records,'local_scientific_witnesses':quality,'oracle':'IndependentAoSFloat64collisionmomentsanddestinationpullstreamingwithretainedoldunwrittenbuffers;sourceGPUissplitlayout/pushstream.','scope':'Two256x32fixtures,eightsteps,bothfullnine-populationbuffers,localcollisionmass/momentumandinteriorobstacles. NoNavierStokesconvergence/closedwall/globalmassguarantee.','precision':'FP64storage/compute;actualomega=Float64(Float321.2)notFloat641.2;hostoracleFP64;artifactFP64.','source_mapping':'Nativeactualgrid256x1,block1x32innamedheight32scope;historicalJuliax256orientationwasexplicitlyaligned,P1phase,notclaimedframeworkspeedup.'}
    out=path.parent/'scientific_moment_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main(sys.argv[1])
