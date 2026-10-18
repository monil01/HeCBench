"""Independent direct correlation and Winograd matrix-expression oracle."""
from pathlib import Path
import struct,json,math,hashlib
P=Path(__file__).resolve().parent;N=131;O=N-2
B=[[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]];AT=[[1,1,1,0],[0,1,-1,-1]]
def transpose(a):return [list(v)for v in zip(*a)]
def matmul(a,b):return [[math.fsum(x*y for x,y in zip(row,col))for col in zip(*b)]for row in a]
def main():
    raw=(P/'fixture_f32.bin').read_bytes();values=struct.unpack('<'+'f'*(len(raw)//4),raw);expected=[];cases=[];at=0
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    g=[[f32(x)for x in row]for row in [[.2,-.3,.4],[.5,.6,.7],[-.8,-.9,.10]]]
    for name in ['source_rand_seed1','signed_dyadic']:
        filt=values[at:at+16];at+=16;image=values[at:at+N*N];at+=N*N;out=[None]*(O*O);direct=[]
        for x in range(O):
            for y in range(O):direct.append(math.fsum(g[i][j]*image[(x+i)*N+y+j]for i in range(3)for j in range(3)))
        for tx in range((O+1)//2):
            for ty in range((O+1)//2):
                tile=[[image[(2*tx+i)*N+2*ty+j] if 2*tx+i<N and 2*ty+j<N else 0 for j in range(4)]for i in range(4)]
                d=matmul(matmul(B,tile),transpose(B));m=[[d[i][j]*filt[4*i+j]for j in range(4)]for i in range(4)];result=matmul(matmul(AT,m),transpose(AT))
                for i in range(2):
                    for j in range(2):
                        x=2*tx+i;y=2*ty+j
                        if x<O and y<O:out[x*O+y]=result[i][j]
        assert all(x is not None for x in out)
        discrepancy=max(abs(x-y)for x,y in zip(out,direct));assert discrepancy<1e-5
        expected+=out
        cases.append({'name':name,'original_filter_F32':g,'transformed_filter_F32':filt,'matrix_vs_direct_correlation_max_abs':discrepancy,'expected_count':len(out)})
    (P/'expected_f64.bin').write_bytes(struct.pack('<'+'d'*len(expected),*expected))
    splits=[{'cpu_offset':p,'cpu_tile_rows':(p*3//100)*32,'gpu_tile_rows':96-(p*3//100)*32,'cpu_copied_rows':min((p*3//100)*64,O),'gpu_launch':p<100}for p in range(101)]
    d={'MAP_SIZE':N,'output_size':O,'cases':cases,'input_count':len(values),'one_split_output_count':O*O,'all_split_output_count':2*101*O*O,'source_filter_generator':'ActualoriginalWinogradConv2D_2x2_filter_transformation fromutils.cu compiledg++O3(noOpenMP) withMAP_SIZE131','source_input_generator':'ActualC srand(1),rand()/(float)RAND_MAX interleavedrowmajor forcase0; independent signed-dyadiccase1','oracle':'Independent Float64BtdB/AtMA matrixproducts usingactualstoredF32transformedfilter, plusdirect3x3correlation withoriginalF32filter','splits':splits,'unique_cpu_tile_rows':[0,32,64,96],'source_scope':'Full101CPUoffsets0..100,source32x8GPUgeometry andrealCPU/GPUoverlap/prefixH2D/fullD2H. NamedMAP_SIZE131, original1024unmeasured. NativeOpenMPpragmasareinactive underoriginalnvccMakefile(no-fopenmp), matchingdeclaredbuild; JuliaCPUloopserial.','input_sha256':hashlib.sha256(raw).hexdigest(),'oracle_sha256':hashlib.sha256((P/'expected_f64.bin').read_bytes()).hexdigest(),'atol':1e-5,'rtol':1e-5}
    (P/'fixture_provenance.json').write_text(json.dumps(d,indent=2)+'\n');print({'outputs':len(expected),'split_outputs':d['all_split_output_count'],'cases':[(c['name'],c['matrix_vs_direct_correlation_max_abs'])for c in cases]})
if __name__=='__main__':main()
