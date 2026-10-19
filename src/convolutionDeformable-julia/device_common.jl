# Source-faithful per-batch (batch_size=1) backward-triplet device mathematics.
# Source itself calls all three kernels with batch_size=1 in its timed backward loop.
@inline function bilinear(im,base,H,W,h,w)
    hl=floor(Int32,h);wl=floor(Int32,w);hh=hl+1;wh=wl+1
    lh=h-Float32(hl);lw=w-Float32(wl);ah=1.0f0-lh;aw=1.0f0-lw
    a=0.0f0;b=0.0f0;c=0.0f0;d=0.0f0
    @inbounds begin
        if hl>=0&&wl>=0;a=im[base+hl*W+wl+17];end
        if hl>=0&&wh<=W-1;b=im[base+hl*W+wh+17];end
        if hh<=H-1&&wl>=0;c=im[base+hh*W+wl+17];end
        if hh<=H-1&&wh<=W-1;d=im[base+hh*W+wh+17];end
    end
    (ah*aw)*a+(ah*lw)*b+(lh*aw)*c+(lh*lw)*d
end
@inline function gradient_weight(h,w,iy,ix,H,W)
    if h<=-1||h>=H||w<=-1||w>=W;return 0.0f0;end
    hl=floor(Int32,h);wl=floor(Int32,w);hh=hl+1;wh=wl+1;weight=0.0f0
    if iy==hl&&ix==wl;weight=(Float32(iy+1)-h)*(Float32(ix+1)-w);end
    if iy==hl&&ix==wh;weight=(Float32(iy+1)-h)*(w+Float32(1-ix));end
    if iy==hh&&ix==wl;weight=(h+Float32(1-iy))*(Float32(ix+1)-w);end
    if iy==hh&&ix==wh;weight=(h+Float32(1-iy))*(w+Float32(1-ix));end
    weight
end
@inline function coordinate_weight(im,base,H,W,h,w,direction)
    if h<=-1||h>=H||w<=-1||w>=W;return 0.0f0;end
    hl=floor(Int32,h);wl=floor(Int32,w);hh=hl+1;wh=wl+1;weight=0.0f0
    @inbounds if direction==0
        if hl>=0&&wl>=0;weight+=-(Float32(wl+1)-w)*im[base+hl*W+wl+17];end
        if hl>=0&&wh<=W-1;weight+=-(w-Float32(wl))*im[base+hl*W+wh+17];end
        if hh<=H-1&&wl>=0;weight+=(Float32(wl+1)-w)*im[base+hh*W+wl+17];end
        if hh<=H-1&&wh<=W-1;weight+=(w-Float32(wl))*im[base+hh*W+wh+17];end
    else
        if hl>=0&&wl>=0;weight+=-(Float32(hl+1)-h)*im[base+hl*W+wl+17];end
        if hl>=0&&wh<=W-1;weight+=(Float32(hl+1)-h)*im[base+hl*W+wh+17];end
        if hh<=H-1&&wl>=0;weight+=-(h-Float32(hl))*im[base+hh*W+wl+17];end
        if hh<=H-1&&wh<=W-1;weight+=(h-Float32(hl))*im[base+hh*W+wh+17];end
    end
    weight
end
@inline function im2col_index!(index,image,offset,mask,columns,p)
    S=p.OH*p.OW;K=p.KH*p.KW;pos=index%S;c=(index÷S)%p.C;group=c÷(p.C÷p.G)
    hin=(pos÷p.OW)*p.SH-p.PH;win=(pos%p.OW)*p.SW-p.PW
    obase=group*2*K*S;mbase=group*K*S;ibase=c*p.H*p.W
    @inbounds for ky=0:p.KH-1,kx=0:p.KW-1
        k=ky*p.KW+kx;h=Float32(hin+ky*p.DH)+offset[obase+2*k*S+pos+17];w=Float32(win+kx*p.DW)+offset[obase+(2*k+1)*S+pos+17];m=mask[mbase+k*S+pos+17];v=0.0f0
        if h>-1&&w>-1&&h<p.H&&w<p.W;v=bilinear(image,ibase,p.H,p.W,h,w);end
        columns[(c*K+k)*S+pos+17]=v*m
    end
    return nothing
end
@inline function col2im_index!(index,columns,offset,mask,grad_image,p)
    S=p.OH*p.OW;K=p.KH*p.KW;k=(index÷S)%K;c=index÷(S*K);pos=index%S;ky=k÷p.KW;kx=k%p.KW;group=c÷(p.C÷p.G)
    obase=group*2*K*S;mbase=group*K*S
    @inbounds begin
        h=Float32((pos÷p.OW)*p.SH-p.PH+ky*p.DH)+offset[obase+2*k*S+pos+17]
        w=Float32((pos%p.OW)*p.SW-p.PW+kx*p.DW)+offset[obase+(2*k+1)*S+pos+17]
        top=columns[index+17]*mask[mbase+k*S+pos+17]
    end
    ih=trunc(Int32,h);iw=trunc(Int32,w)
    for dy=-2:2,dx=-2:2
        yy=ih+dy;xx=iw+dx
        if 0<=yy<p.H&&0<=xx<p.W&&abs(h-Float32(yy))<1&&abs(w-Float32(xx))<1
            weight=gradient_weight(h,w,yy,xx,p.H,p.W)
            atomic_add_pixel!(grad_image,(c*p.H+yy)*p.W+xx+17,weight*top)
        end
    end
    return nothing
end
@inline function coord_index!(index,columns,image,offset,mask,grad_offset,grad_mask,p)
    S=p.OH*p.OW;K=p.KH*p.KW;pos=index%S;c=(index÷S)%(p.G*2*K);group=c÷(2*K);offc=c-group*2*K;channel_columns=p.C*K÷p.G
    cb=group*channel_columns*S;ib=group*(p.C÷p.G)*p.H*p.W;ob=group*2*K*S;mb=group*K*S
    val=0.0f0;mval=0.0f0;cnt=Int32(0)
    @inbounds for colc=(offc÷2):K:channel_columns-1
        k=colc%K;ky=k÷p.KW;kx=k%p.KW;colpos=colc*S+pos;direction=offc%2
        h=Float32((pos÷p.OW)*p.SH-p.PH+ky*p.DH)+offset[ob+2*k*S+pos+17]
        w=Float32((pos%p.OW)*p.SW-p.PW+kx*p.DW)+offset[ob+(2*k+1)*S+pos+17]
        m=mask[mb+k*S+pos+17];g=columns[cb+colpos+17];base=ib+cnt*p.H*p.W
        if h<=-1||w<=-1||h>=p.H||w>=p.W
            h=-2.0f0;w=-2.0f0
        else
            mval+=g*bilinear(image,base,p.H,p.W,h,w)
        end
        weight=coordinate_weight(image,base,p.H,p.W,h,w,direction)
        val+=(weight*g)*m;cnt+=1
    end
    @inbounds grad_offset[index+17]=val
    if offc%2==0;@inbounds grad_mask[(group*K+offc÷2)*S+pos+17]=mval;end
    return nothing
end
