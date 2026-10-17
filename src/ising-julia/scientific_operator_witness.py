#!/usr/bin/env python3
"""Audit every saved spin via physical-grid bonds, independent of packed kernels."""
import argparse,hashlib,json,math,struct
from pathlib import Path

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def fixture(nx,ny,alpha,sweeps):
    u=[f32((((37*i+11)%65536)+.5)/65536) for i in range(nx*(ny//2))];u[:3]=[0.,.5,1.]
    beta=f32(1/f32(f32(alpha)*f32(2.26918531421)))
    return u,beta

def physical(nx,ny,u,beta,steps):
    g=[[(-1 if u[i*(ny//2)+j//2]<.5 else 1) for j in range(ny)] for i in range(nx)]
    for _ in range(steps):
        for color in (0,1):
            for i in range(nx):
                for j in range(ny):
                    if (i+j)%2!=color:continue
                    delta=2*g[i][j]*(g[(i-1)%nx][j]+g[(i+1)%nx][j]+g[i][(j-1)%ny]+g[i][(j+1)%ny])
                    if u[i*(ny//2)+j//2]<math.exp(-beta*delta):g[i][j]*=-1
    packed=[g[i][2*j+(i%2 if color==0 else 1-i%2)] for color in (0,1) for i in range(nx) for j in range(ny//2)]
    mag=sum(map(sum,g));energy=-sum(g[i][j]*(g[(i+1)%nx][j]+g[i][(j+1)%ny]) for i in range(nx) for j in range(ny))
    assert abs(mag)<=nx*ny and abs(energy)<=2*nx*ny and mag%2==0
    return packed,[mag,energy]

def main():
    a=argparse.ArgumentParser();a.add_argument('result',type=Path);a.add_argument('--output',type=Path);args=a.parse_args()
    data=json.loads(args.result.read_text());prefix=[];spins=[];obs=[];descriptors=[]
    for nx,ny,alpha,sweeps in ((18,34,.5,17),(32,66,1.,18),(2,4,2.,19)):
        u,beta=fixture(nx,ny,alpha,sweeps);prefix += [nx,ny,sweeps,alpha,beta]+u
        phases=[]
        for steps in (0,2,sweeps+2):
            v,o=physical(nx,ny,u,beta,steps);spins+=v;phases.append({'sweeps':steps,'magnetization':o[0],'energy':o[1]})
        obs+=o
        gaps=[abs(x-math.exp(-beta*d)) for x in u for d in (-8,-4,0,4,8) if x!=math.exp(-beta*d)]
        descriptors.append({'nx':nx,'ny':ny,'alpha':alpha,'beta_float32':beta,'minimum_nontie_decision_gap':min(gaps),'phases':phases})
    expected=struct.pack('<9583f',*(prefix+spins+obs));records=[]
    assert len(prefix)==1381 and len(spins)==8196 and len(obs)==6
    for model,item in data['models'].items():
        for row in item['trials']:
            artifact=row['output_artifact'];p=Path(artifact['saved_path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            assert digest==artifact['sha256'] and raw==expected,(model,row['trial'])
            records.append({'model':model,'trial':row['trial'],'artifact_sha256':digest,'exact_physical_oracle':True})
    assert len(records)==20
    result={'benchmark':'ising','result':str(args.result),'full_artifacts_audited':20,'computed_spin_values_per_artifact':8196,'prefix_values':1381,'derived_observables':6,'physical_grid_oracle':'Float64 exp with actual Float32 beta; literal 2D periodic bonds, checkerboard parity, strict comparison including threshold1; no packed-neighbor formula','descriptors':descriptors,'records':records,'limitations':['Fixed reused thresholds implement source verification mode, not fresh RNG or equilibrium statistics','Three small named grids; no finite-size scaling or critical-temperature inference','Magnetization and energy are derived observables, not extra GPU-computed validation values']}
    output=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(output)
    print(output)
if __name__=='__main__':main()
