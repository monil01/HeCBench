# GFC benchmark-copy codec corrected to initialize sign/count and use warp phase barriers.
# Original copyright/license in ../fpdc-cuda/main.cu; exact minimal native patch retained.
function fpdc_lane!(dim,src,dst,cuts,offsets,scratch,tid::Int32,global0::Int32)
    lane=tid&Int32(31);warp=global0÷Int32(32);iw=(tid÷Int32(32))*Int32(48)+lane
    @inbounds scratch[iw+1]=Int32(0)
    CUDA.sync_warp()
    iw+=Int32(16);last=(tid÷Int32(32)+Int32(1))*Int32(48)-Int32(1)
    predictor_offset=Int32(32)-(dim-lane%dim)-lane
    start=warp==0 ? Int32(0) : (@inbounds cuts[warp]);term=@inbounds cuts[warp+1]
    outpos=((start+Int32(1))÷Int32(2))*Int32(17);previous=UInt64(0);i=start+lane
    while i<term
        current=@inbounds src[i+1]
        code=current<previous ? Int32(8) : Int32(0)
        difference=current>=previous ? current-previous : previous-current
        count=Int32(8-(leading_zeros(difference)>>3));count==2&&(count=Int32(3))
        @inbounds scratch[iw+1]=count
        CUDA.sync_warp()
        for delta in (1,2,4,8,16)
            next=(@inbounds scratch[iw+1])+(@inbounds scratch[iw-delta+1])
            CUDA.sync_warp()
            @inbounds scratch[iw+1]=next
            CUDA.sync_warp()
        end
        beginpos=outpos+Int32(16)+(@inbounds scratch[iw]);endpos=beginpos+count
        while beginpos<endpos
            @inbounds dst[beginpos+1]=UInt8(difference&UInt64(255))
            difference>>=8;beginpos+=Int32(1)
        end
        count>=3&&(count-=Int32(1))
        total=@inbounds scratch[last+1]
        CUDA.sync_warp()
        code|=count
        @inbounds scratch[iw+1]=code
        CUDA.sync_warp()
        if (lane&Int32(1))!=0
            packed=(@inbounds scratch[iw])|(code<<4)
            @inbounds dst[outpos+(lane>>1)+1]=UInt8(packed&Int32(255))
        end
        outpos+=total+Int32(16)
        previous=@inbounds src[i+predictor_offset+1]
        CUDA.sync_warp();i+=Int32(32)
    end
    if lane==31;@inbounds offsets[warp+1]=outpos;end
    nothing
end
