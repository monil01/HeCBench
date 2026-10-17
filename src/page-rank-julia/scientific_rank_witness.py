"""Independent sparse Markov oracle and stationarity diagnostics, outside GPU timing."""
import array,json,math,sys
from pathlib import Path
N=257;ITERS=64;PER=N*N+ITERS*N+N
f32=lambda x:array.array('f',[x])[0]
ALPHA=f32(.85);INITIAL=f32(1/N);TELEPORT=f32(f32(1-ALPHA)/N)
def graph(irregular):
    result=[]
    for i in range(N):
        targets={(i+1)%N,(i+13)%N,(17*i+5)%N} if irregular else {(i+1)%N,(i+13)%N,(i+37)%N}
        if irregular:
            if i%3==0:targets.add((i+37)%N)
            if i!=0:targets.add(0)
            targets.discard(i)
        result.append(sorted(targets))
    return result
def push(rank,edges,alpha=ALPHA,teleport=TELEPORT):
    incoming=[0.]*N
    for value,targets in zip(rank,edges):
        for j in targets:incoming[j]+=value/len(targets)
    return [teleport+alpha*x for x in incoming]
def reference(edges):
    rank=[INITIAL]*N;history=[0.]*N;trajectory=[]
    for _ in range(ITERS):
        old=rank;rank=push(old,edges);history=[max(h,abs(a-b)) for h,a,b in zip(history,rank,old)];trajectory.extend(rank)
    maps=[old[i]/len(edges[i]) if j in edges[i] else 0. for i in range(N) for j in range(N)]
    return maps+trajectory+history

def main(result_path):
    result_path=Path(result_path);r=json.loads(result_path.read_text());graphs=[graph(False),graph(True)];expected=[reference(g) for g in graphs];records=[]
    for model,data in r['models'].items():
        if len(data.get('trials',[]))!=5:raise AssertionError((model,'not five trials'))
        for trial in data['trials']:
            a=array.array('f');a.frombytes(Path(trial['output_artifact']['saved_path']).read_bytes());assert len(a)==2*PER
            for case,g in enumerate(graphs):
                values=a[case*PER:(case+1)*PER];err=max(abs(x-y) for x,y in zip(values,expected[case]));assert err<=1e-6
                trajectory=values[N*N:N*N+N*ITERS];rank=list(trajectory[-N:]);previous=trajectory[-2*N:-N];history=values[-N:]
                mass_error=max(abs(math.fsum(trajectory[k*N:(k+1)*N])-1) for k in range(ITERS));minimum=min(trajectory);assert mass_error<5e-6 and minimum>0
                assert all(history[j]>=abs(rank[j]-previous[j])-1e-8 for j in range(N))
                ideal=push(rank,g,.85,.15/N);residual=math.fsum(abs(x-y) for x,y in zip(rank,ideal))
                stationary=[1/N]*N
                for k in range(10000):
                    new=push(stationary,g,.85,.15/N)
                    if max(abs(x-y) for x,y in zip(new,stationary))<1e-16:stationary=new;break
                    stationary=new
                distance=math.fsum(abs(x-y) for x,y in zip(rank,stationary));bound=residual/.15;assert distance<=bound+1e-12
                if case==1:assert max(rank)-min(rank)>0.01
                records.append({'model':model,'trial':trial['trial'],'graph':case,'max_abs_sparse_oracle_error':err,'max_probability_mass_error':mass_error,'minimum_rank':minimum,'ideal_pagerank_l1_residual':residual,'ideal_stationary_l1_error':distance,'l1_error_upper_bound_residual_over_1_minus_alpha':bound,'current_max_step_change':max(abs(x-y) for x,y in zip(rank,previous)),'historical_max_step_change':max(history),'degree_range':[min(map(len,g)),max(map(len,g))]})
    report={'result_path':str(result_path.resolve()),'all_models_trials_graphs':records,'checked_values_per_trial':2*PER,'checked_artifact_count':sum(len(x['trials']) for x in r['models'].values()),'oracle':'Independent sparse outgoing-edge Float64 Markov push, not dense GPU column reduction; all final maps, all64 rankvectors, history.','stationarity':'For ideal alpha=.85 exactly and uniform teleportation, L1 stochastic contraction gives error <= residual/(1-alpha). Actual GPU constants are Float32; ideal versus source-constant distinction is explicit.','scope':'Two named non-dangling graphs,257 vertices,64 fixed iterations. No claim historical diagnostic is convergence residual or default native generator is initialized.','not_rechecked':'Intermediate dense maps except final map; dangling vertices; adaptive stopping; arbitrary sizes/graphs.'}
    output=result_path.parent/'scientific_rank_witness.json';output.write_text(json.dumps(report,indent=2)+'\n');print(output)
if __name__=='__main__':main(sys.argv[1])
