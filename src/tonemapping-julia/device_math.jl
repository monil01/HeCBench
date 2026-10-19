@inline tone_lum(r,g,b)=(.2126f0*r+.7152f0*g)+.0722f0*b
@inline function tone_pixel!(x,y,input,output,av,gamma,c,delta,w,h)
    @inbounds begin
        index=4*(y*w+x)+1
        r=input[index];g=input[index+1];b=input[index+2]
        l=tone_lum(r,g,b);mean=l
        if x!=0 && y!=0 && x!=w-1 && y!=h-1
            mean=0f0
            for dy in -1:1, dx in -1:1
                if dx!=0 || dy!=0
                    j=4*((y+dy)*w+x+dx)+1
                    mean+=tone_lum(input[j],input[j+1],input[j+2])
                end
            end
            mean/=8f0
        end
        cp=mean*log(delta+mean/l)+c*av
        display=l/(l+cp)
        output[index]=(r/l)^gamma*display
        output[index+1]=(g/l)^gamma*display
        output[index+2]=(b/l)^gamma*display
        output[index+3]=input[index+3]
    end
    nothing
end
@inline function tone_echo!(p,av,gamma,c,delta,w,h)
    @inbounds begin p[1]=av;p[2]=gamma;p[3]=c;p[4]=delta;p[5]=Float32(w);p[6]=4f0;p[7]=Float32(h);end
    nothing
end
@inline function tone_coords(index,w)
    lane=index-1;group=lane÷256;tx=lane%16;ty=(lane÷16)%16
    bx=group%(w÷16);by=group÷(w÷16)
    (bx*16+tx,by*16+ty)
end
