from pathlib import Path
import struct,json,math
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
FX,FY,NA,NC,NB=17,19,6,3,7
K=FX*FY*NA;G=f32(-12345)
limits=list(map(f32,[0,69.12,-39.68,39.68]));offset=f32(.78539);thresholds=list(map(f32,[.1,.5,1]))
anchors=list(map(f32,[3.9,1.6,1.56,0,3.9,1.6,1.56,1.57,.8,.6,1.73,0,.8,.6,1.73,1.57,1.76,.6,1.73,0,1.76,.6,1.73,1.57]));bottom=list(map(f32,[-1.78,-.6,-.6]));cls=[];box=[];dirs=[];period=f32(math.pi)
patterns=[[-8,-8,-8],[-4,-3,-2],[0,0,-1],[1,1,1],[-1,2,-1],[-1,-2,3],[0,-5,-5],[-2,-3,-4]]
for i in range(K):
 a=i%NA;cls+=list(map(f32,patterns[i%8]));vals=[f32(((i*7+j*11)%65-32)/16) for j in range(3)]+[f32(((i*13+j*5)%33-16)/16) for j in range(3)]
 if i%11==0 and a%2==0:yaw=offset
 elif i%13==0:yaw=f32(offset-anchors[4*a+3]+((i%7)-3)*period+(-.0001 if i%2 else .0001))
 else:yaw=f32([-3,-1,.2,2.3,5.1][i%5])
 box+=vals+[yaw]
 dirs+=list(map(f32,[0,0] if i%3==0 else [1,-1] if i%3==1 else [-1,1]))
box+=[G]*17
prefix=list(map(f32,[FX,FY,NA,NC,NB]))+limits+[offset]+anchors+bottom+thresholds+cls+box+dirs
expected=[];descriptors=[]
for thresh in thresholds:
 changed=list(map(float,box));bnd=[float(G)]*(K*9+17);scores=[float(G)]*(K+17);accepted=[];class_hist=[0]*NC;ties=0;direction_ties=0
 for i in range(K):
  loc,a=divmod(i,NA);col,row=loc%FX,loc//FX;logits=cls[i*NC:(i+1)*NC];probs=[1/(1+math.exp(-x)) for x in logits];cl=max(range(NC),key=lambda k:probs[k]);score=probs[cl]
  if score<thresh:continue
  class_hist[cl]+=1;ties+=sum(x==max(logits) for x in logits)>1;direction_ties+=dirs[2*i]==dirs[2*i+1];base=i*NB;dxa,dya,dza,ra=anchors[4*a:4*a+4];xa=limits[0]+col*(limits[1]-limits[0])/(FX-1);ya=limits[2]+row*(limits[3]-limits[2])/(FY-1);za=dza/2+bottom[a//2];diag=math.sqrt(dxa*dxa+dya*dya);out=[box[base]*diag+xa,box[base+1]*diag+ya,box[base+2]*dza+za,math.exp(box[base+3])*dxa,math.exp(box[base+4])*dya,math.exp(box[base+5])*dza,box[base+6]+ra];changed[base:base+7]=out;label=0 if dirs[2*i]>dirs[2*i+1] else 1;val=out[6]-offset;yaw=val-math.floor(val/period)*period+offset+period*label;j=len(accepted);accepted.append(i);bnd[j*9:j*9+9]=out[:6]+[yaw,float(cl),float(base)];scores[j]=score
 expected+=[float(len(accepted))]+changed+bnd+scores
 descriptors.append({'threshold':thresh,'accepted':len(accepted),'rejected':K-len(accepted),'class_histogram':class_hist,'accepted_tied_class_maxima':ties,'accepted_direction_ties':direction_ties,'accepted_source_anchor_ids':accepted})
(p/'fixture_bits.bin').write_bytes(struct.pack('<'+'f'*len(prefix),*prefix));(p/'expected_full_outputs.bin').write_bytes(struct.pack('<'+'d'*len(expected),*expected))
(p/'fixture_definition.json').write_text(json.dumps({'feature_x':FX,'feature_y':FY,'anchors_per_location':NA,'total_anchors':K,'input_prefix_values':len(prefix),'per_path_public_values':1+(K*NB+17)+(K*9+17)+(K+17),'physical_outputs':len(expected),'guards':17,'guard_value':G,'threshold_cases':descriptors,'generation':'Deterministic dyadictranslation/logscaleencodings, variedclasslogits inclstrictmaxtie/exact0scoreboundary, directionties, negativeandpositiveyaw/wrap±1e-4; exactwrap0onlyanchorswithrotation0.','oracle':'IndependentFloat64sigmoid/exp/sqrt/boxtransform andyawmoduloFloat32period, allrecords/sourceIDs/pairedscores/fullmutatedboxes; rejectedboxesandallunused/physicalguards exact. Canonicalizeonlyactualunorderedrecords bysourceboxoffset, retainraworderfullsidecars.','tolerances':{'decoded_coordinates_yaw':.001,'scores':.000001,'IDs_counts_rejected_values_guards':'exact'},'named_scope':'Small17x19 featuregrid/source6-threadblocks, threshold.1/.5/1; notfull216x248defaultoractualLiDARdetectionaccuracy.'},indent=2)+'\n')
print(K,len(prefix),len(expected),[x['accepted'] for x in descriptors])
