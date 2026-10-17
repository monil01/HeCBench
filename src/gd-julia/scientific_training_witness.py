"""Independent sparse logistic loss/gradient, optimization, and full-state witness."""
from pathlib import Path
import array,json,math,sys
f32=lambda v:array.array('f',[v])[0]
M=257;N=512;ALPHA=f32(.4);LAMBDA=f32(.01)
def fixture():
    rows=[0];cols=[];vals=[];labels=[]
    for i in range(M):
        y=-1 if i%2 else 1;labels.append(y);cols.extend([0,1+(37*i)%511,1+(37*i+13)%511,1+(37*i+211)%511]);vals.extend([.5*y,-.5,.5,.5]);rows.append(len(cols))
    return rows,cols,vals,labels

def evaluate(x,rows,cols,vals,labels):
    grad=[0.]*N;loss=0.;correct=0
    for i in range(M):
        xp=math.fsum(vals[j]*x[cols[j]]for j in range(rows[i],rows[i+1]));y=labels[i];loss+=math.log1p(math.exp(-y*xp));correct+=((1 if xp>=0 else -1)==y)
        factor=-y/(1+math.exp(y*xp))
        for j in range(rows[i],rows[i+1]):grad[cols[j]]+=factor*vals[j]
    l2=math.fsum(v*v for v in x);return loss,grad,l2,correct

def main(path):
    path=Path(path);r=json.loads(path.read_text());rows,cols,vals,labels=fixture();x=[0.]*N;losses=[];fd=[]
    assert all(math.fsum(vals[j]**2 for j in range(rows[i],rows[i+1]))==1 for i in range(M))
    for iteration in range(32):
        previous=x.copy();loss,grad,l2,correct=evaluate(x,rows,cols,vals,labels);losses.append(loss/M+.5*LAMBDA*l2)
        if iteration in (0,31):
            for coordinate in (0,13,511):
                h=1e-5;plus=x.copy();minus=x.copy();plus[coordinate]+=h;minus[coordinate]-=h
                p,_,lp,_=evaluate(plus,rows,cols,vals,labels);m,_,lm,_=evaluate(minus,rows,cols,vals,labels);numerical=((p/M+.5*LAMBDA*lp)-(m/M+.5*LAMBDA*lm))/(2*h);analytic=grad[coordinate]/M+LAMBDA*x[coordinate];assert abs(numerical-analytic)<1e-8
                fd.append({'iteration':iteration,'coordinate':coordinate,'analytic_gradient':analytic,'central_difference_gradient':numerical,'abs_error':abs(numerical-analytic)})
        x=[v-ALPHA*(g/M+LAMBDA*v)for v,g in zip(x,grad)]
    assert all(b<=a+1e-12 for a,b in zip(losses,losses[1:]));assert losses[-1]<losses[0]-.1 and x[0]>.1 and correct==M
    # Unit-norm rows bound the average data Hessian norm by1/4, plus L2 lambda.
    assert ALPHA<2/(.25+LAMBDA)
    prefix=array.array('f',rows+cols+vals+labels+[LAMBDA,ALPHA,32]+[0.]*N);expected=x+grad+[loss,l2,float(correct)];records=[]
    for model,data in r['models'].items():
        assert len(data['trials'])==5
        for trial in data['trials']:
            raw=Path(trial['output_artifact']['saved_path']).read_bytes();assert raw[:3086*4]==prefix.tobytes();out=array.array('f');out.frombytes(raw);assert len(out)==4113;got=out[3086:]
            assert all(math.isfinite(v)and abs(v-w)<=1e-4+1e-3*abs(w)for v,w in zip(got,expected));assert got[-1]==correct
            records.append({'model':model,'trial':trial['trial'],'exact_input_prefix_bytes':12344,'weights_max_abs_error':max(abs(v-w)for v,w in zip(got[:512],x)),'gradient_max_abs_error':max(abs(v-w)for v,w in zip(got[512:1024],grad)),'objective_abs_error':abs(got[1024]-loss),'l2_abs_error':abs(got[1025]-l2),'correct_count_exact':True})
    report={'result_path':str(path.resolve()),'all_model_trials':records,'finite_difference_gradient_checks':fd,'regularized_loss_cpu_trajectory':losses,'initial_loss':losses[0],'final_preupdate_loss':losses[-1],'final_correct':correct,'final_signal_weight':x[0],'normalized_rows':True,'gradient_step_sufficient_descent_bound':{'lipschitz_upper_bound':.25+LAMBDA,'step':ALPHA,'step_less_than_two_over_bound':True},'oracle':'IndependentFloat64log1p/sigmoidmathematicalgradient,notGPUexp-ratioformula;centralfinite-differencesactualregularizedobjective.','state_alignment':'Finalweightsafter32updates;gradient/objective/L2/classificationfrompre-updatex31,matchingnativekernelorder. CPUtrajectorychecked,intermediateGPUstatesnotexported.','scope':'Synthetic257x512unitnormCSRfixture,512featuresfull256threadnormblocks. No arbitrarypartialCUBcollectiveclaim orreal-gisettedatasetaccuracycertification. NativeCUBvsmanualsharedtreeandCUDAmemsetvsJuliafill resource/hostAPIadaptationdeclared.'}
    out=path.parent/'scientific_training_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main(sys.argv[1])
