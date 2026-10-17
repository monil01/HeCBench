#=
/* -------------------------------------------------------------------------- */
/* Rainflow cycle counting algorithm according to:                            */
/* ASTM E1049-85,                                                             */
/* Standard Practices for Cycle Counting in Fatigue Analysis,                 */
/* ASTM International. (https://doi.org/10.1520/E1049-85R17)                  */
/*                                                                            */
/* By: Carlos Souto - csouto@fe.up.pt                                         */
/* -------------------------------------------------------------------------- */
/* Extracted from ASTM E1049-85 – Rainflow Counting:                          */
/* Rules for this method are as follows: let X denote range under             */
/* consideration; Y, previous range adjacent to X; and S, starting point in   */
/* the history.                                                               */
/* (1) Read next peak or valley. If out of data, go to Step 6.                */
/* (2) If there are less than three points, go to Step 1. Form ranges X and Y */
/*     using the three most recent peaks and valleys that have not been       */
/*     discarded.                                                             */
/* (3) Compare the absolute values of ranges X and Y.                         */
/*     (a) If X < Y, go to Step 1.                                            */
/*     (b) If X >= Y, go to Step 4.                                           */
/* (4) If range Y contains the starting point S, go to Step 5; otherwise,     */
/*     count range Y as one cycle; discard the peak and valley of Y; and go   */
/*     to Step 2.                                                             */
/* (5) Count range Y as one-half cycle; discard the first point (peak or      */
/*     valley) in range Y; move the starting point to the second point in     */
/*     range Y; and go to Step 2.                                             */
/* (6) Count each range that has not been previously counted as one-half      */
/*     cycle.                                                                 */
/* -------------------------------------------------------------------------- */
/* License: BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause       */
/*                                                                            */
/* Copyright (c) 2021, Carlos Daniel Santos Souto.                            */
/* All rights reserved.                                                       */
/*                                                                            */
/* Redistribution and use in source and binary forms, with or without         */
/* modification, are permitted provided that the following conditions are     */
/* met:                                                                       */
/*                                                                            */
/* 1. Redistributions of source code must retain the above copyright notice,  */
/*    this list of conditions and the following disclaimer.                   */
/* 2. Redistributions in binary form must reproduce the above copyright       */
/*    notice, this list of conditions and the following disclaimer in the     */
/*    documentation and/or other materials provided with the distribution.    */
/* 3. Neither the name of the copyright holder nor the names of its           */
/*    contributors may be used to endorse or promote products derived from    */
/*    this software without specific prior written permission.                */
/*                                                                            */
/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS    */
/* IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,  */
/* THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR     */
/* PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          */
/* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      */
/* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        */
/* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         */
/* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     */
/* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       */
/* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         */
/* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               */
/* -------------------------------------------------------------------------- */
=#
@inline function rainflow_eval!(id,history,offsets,extrema,points,results,lengths)
    @inbounds base=offsets[id];len=offsets[id+1]-base;extrema[base+Int32(1)]=history[base+Int32(1)]
    eidx=Int32(0)
    @inbounds for i in Int32(1):len-Int32(2)
        v=history[base+i+Int32(1)];previous=extrema[base+eidx+Int32(1)];next=history[base+i+Int32(2)]
        if (v>previous&&v>next)||(v<previous&&v<next);eidx+=Int32(1);extrema[base+eidx+Int32(1)]=v;end
    end
    eidx+=Int32(1);@inbounds extrema[base+eidx+Int32(1)]=history[base+len];elen=eidx+Int32(1);pidx=Int32(-1);eidx=Int32(-1);ridx=Int32(-1)
    @inbounds for i in Int32(0):elen-Int32(1)
        pidx+=Int32(1);eidx+=Int32(1);points[base+pidx+Int32(1)]=eidx
        while pidx>=Int32(2)
            pa=points[base+pidx-Int32(1)];pb=points[base+pidx];pc=points[base+pidx+Int32(1)]
            x=abs(extrema[base+pb+Int32(1)]-extrema[base+pc+Int32(1)]);y=abs(extrema[base+pa+Int32(1)]-extrema[base+pb+Int32(1)])
            x<y&&break;mean=0.5*(extrema[base+pa+Int32(1)]+extrema[base+pb+Int32(1)]);ridx+=Int32(1);q=Int32(3)*(base+ridx)
            results[q+Int32(1)]=pidx==Int32(2) ? 0.5 : 1.0;results[q+Int32(2)]=y;results[q+Int32(3)]=mean
            if pidx==Int32(2)
                points[base+Int32(1)]=points[base+Int32(2)];points[base+Int32(2)]=points[base+Int32(3)];pidx=Int32(1)
            else
                points[base+pidx-Int32(1)]=points[base+pidx+Int32(1)];pidx-=Int32(2)
            end
        end
    end
    @inbounds for i in Int32(0):pidx-Int32(1)
        va=extrema[base+points[base+i+Int32(1)]+Int32(1)];vb=extrema[base+points[base+i+Int32(2)]+Int32(1)];ridx+=Int32(1);q=Int32(3)*(base+ridx);results[q+Int32(1)]=0.5;results[q+Int32(2)]=abs(va-vb);results[q+Int32(3)]=0.5*(va+vb)
    end
    @inbounds lengths[id]=ridx+Int32(1)
    nothing
end
