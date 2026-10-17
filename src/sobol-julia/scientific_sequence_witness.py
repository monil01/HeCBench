"""Independent polynomial+Graycode oracle and scoped QMC sanity witnesses."""
from pathlib import Path
import re,json,array,struct,math,sys
CASES=[(1,7),(257,7),(16387,7),(1027,100)]
def directions():
    src=Path(__file__).parent.parent/'sobol-cuda/sobol_primitives.cu';rows=[]
    for m in re.finditer(r'\{(\d+),\s*(\d+),\s*(\d+),\s*\{([^}]*)\}\}',src.read_text()):
        rows.append((int(m[2]),int(m[3]),[int(x)for x in m[4].split(',')if x.strip()]))
    out=[]
    for dim,(degree,polynomial,initial)in enumerate(rows[:100]):
        if dim==0:v=[1<<(31-i)for i in range(32)]
        else:
            v=[initial[i]<<(31-i)for i in range(degree)]
            for i in range(degree,32):
                x=v[i-degree]^(v[i-degree]>>degree)
                for j in range(1,degree):
                    if polynomial&(1<<(degree-1-j)):x^=v[i-j]
                v.append(x)
        out.extend(v)
    return out

def outputs(n,d,dirs):
    ret=array.array('I')
    for dim in range(d):
        for i in range(n):
            gray=i^(i>>1);x=0
            for bit in range(32):
                if gray&(1<<bit):x^=dirs[dim*32+bit]
            # Integer conversion rounds to binary32 first, then exact power-of-two scaling.
            rounded=struct.unpack('<f',struct.pack('<f',float(x)))[0]
            ret.append(struct.unpack('<I',struct.pack('<f',rounded*2**-32))[0])
    return ret

def main(path):
    path=Path(path);r=json.loads(path.read_text());dirs=directions();want=array.array('I',dirs)
    expected=[];quality=[]
    for n,d in CASES:
        bits=outputs(n,d,dirs);expected.append(bits);want.extend(bits);values=array.array('f');values.frombytes(bits.tobytes());assert all(0<=x<1 for x in values)
        if n>=1024:
            net=1<<int(math.log2(n));assert all(len({int(values[dim*n+i]*net)for i in range(net)})==net for dim in range(d))
            integral=math.fsum(values[i]*values[n+i]for i in range(net))/net;old=math.fsum(values[i]**2 for i in range(net))/net
            assert abs(integral-.25)<.002 and abs(old-.25)>.08
            assert any(values[i]!=values[n+i]for i in range(net))
            quality.append({'vectors':n,'dimensions':d,'dyadic_prefix':net,'all_dimensions_one_point_per_dyadic_bin':True,'mean_x1_x2':integral,'known_integral_x1_x2':.25,'old_repeated_dimension_mean_x_squared':old,'old_shortcut_rejected':True,'scope':'Namedintegrandand1Dstratificationonly,notgeneralQMCerrorrateorallhighdimensionalprojections.'})
    assert len(want)==222415;raw=want.tobytes();records=[]
    for model,m in r['models'].items():
        assert len(m['trials'])==5
        for trial in m['trials']:
            got=Path(trial['output_artifact']['saved_path']).read_bytes();assert got[:12800]==raw[:12800];assert got==raw;records.append({'model':model,'trial':trial['trial'],'input_prefix_bytes':12800,'output_float32_values':219215,'bitwise_equal':True})
    report={'result_path':str(path.resolve()),'all_model_trials':records,'quality_witnesses':quality,'oracle':'IndependenthostGF(2)polynomialrecurrence+direct32bitGraycodeXOR,notGPUstridedincrementalrecurrence;Float32roundingexplicit','scientific_reference_repair':'Actualfirst100Joe-Kuoprimitivedirectionrowsreplacerepeateddimension1initialization. OriginalGPUkernelpreserved. HistoricalselfCPUagreementwithsamerepeatedtablewasinsufficient.','scope':'4namedshapesincludingbothgridpolicybranches/tails/gridstride;first100of10200dimensionsonly;noall-dimensionstatisticalcertification.'}
    out=path.parent/'scientific_sequence_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main(sys.argv[1])
