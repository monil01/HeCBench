from pathlib import Path
from fractions import Fraction as F
import struct,json
p=Path(__file__).resolve().parent;pre=[24];gold=[];paths=[]
shapes=[(8,6,4,3),(9,7,6,4),(12,14,24,28),(5,3,5,3)]
for bits in [8,16,32]:
 mask=(1<<bits)-1;lead=256//(bits//8);guard={8:0xa5,16:0xa55a,32:0xa55aa55a}[bits]
 for shape,(iw,ih,ow,oh) in enumerate(shapes,1):
  for mode in [0,1]:
   C=16;repeat=3;h=[];out=[]
   for c in range(C):
    for y in range(ih):
     for x in range(iw):
      if mode:
       factor=1 if bits==8 else 16
       value=7 if c==0 else (11+3*x+5*y if c==1 else 8*c+3*x+5*y+(x+y+c)%3);value*=factor
      else:
       constants=[0,mask,1,1<<(bits-1),(1<<min(bits-1,24))-1,min((1<<24)+1,mask)]
       value=constants[c] if c<6 else ((c*99991+y*1013+x*2654435761+17)&mask)
      assert 0<=value<=mask;h.append(value)
   for c in range(C):
    def sample(x,y):return h[(c*ih+y)*iw+x]
    for y in range(oh):
     for x in range(ow):
      if not mode:
       xx=min((F(2*x+1,2)*F(iw,ow)+F(1,2)).numerator//(F(2*x+1,2)*F(iw,ow)+F(1,2)).denominator,iw-1);yy=min((F(2*y+1,2)*F(ih,oh)+F(1,2)).numerator//(F(2*y+1,2)*F(ih,oh)+F(1,2)).denominator,ih-1);value=sample(xx,yy)
      else:
       xx=max(F(2*x+1,2)*F(iw,ow)-F(1,2),0);yy=max(F(2*y+1,2)*F(ih,oh)-F(1,2),0);x0=xx.numerator//xx.denominator;y0=yy.numerator//yy.denominator;u=xx-x0;v=yy-y0;x1=min(x0+1,iw-1);y1=min(y0+1,ih-1)
       # Independent convex-weight form, not native expanded cross term.
       value=(1-u)*(1-v)*sample(x0,y0)+u*(1-v)*sample(x1,y0)+(1-u)*v*sample(x0,y1)+u*v*sample(x1,y1);value=value.numerator//value.denominator
      out.append(value)
   pre.extend([bits,mode,iw,ih,ow,oh,C,repeat]);pre.extend(h);gold.extend(out);gold.extend(h);gold.extend([guard]*(2*(lead+16)))
   paths.append(dict(path=len(paths)+1,bits=bits,mode='bilinear' if mode else 'nearest',iw=iw,ih=ih,ow=ow,oh=oh,channels=C,repeat=repeat,input_values=len(h),output_values=len(out),guard_fields=2*(lead+16),max_input=max(h),min_input=min(h),scales=[float(F(iw,ow)),float(F(ih,oh))],shape=shape))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'oracle.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold));(p/'fixture_description.json').write_text(json.dumps(dict(paths=paths,input_prefix_values=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),oracle='Independent Fraction convex four-corner bilinear weights/floorUIntquantization; nearest retains source halfcenter+positiveawayties/upperclamp. All scales dyadic, boundedbilinearUInt8<=220/UInt16-32<=3520; every polynomial intermediate numerator fits24bitFloat32 exactly.',source_coordinate_convention='Nearest(y+.5)*scale withpositiveawayties andclamp, no-.5; identitynearestshiftsonepixelbydeclaredsourceconvention. Bilinearmax((y+.5)*scale-.5,0),clampedupperneighbor,truncateoutputtowardzero.',storage_coverage='UInt8/16/32fullrange nearestcopy, UInt16/32bilinear restrictedlowrange forFloat32exact rational domain; highrange32bitinterpolationnotcertified.',geometry=dict(blocks=29184,threads=256,channels_per_iter=8,mostly_inactive_grid_for_named_small_shapes=True),unverified='Nonmultiple8channeltail,falsecoordinateflags,non-dyadicratios,Float32precisionlossforhighUInt32bilinear,sourcefull1920x1080→256x256channels512CLIimagequality/throughput.'),indent=2)+'\n');print(len(pre),len(gold),len(pre)+len(gold))
