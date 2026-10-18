# Derived from EUPL-1.1 EasyWave kernels.cuh, Babeyko/Spazier2014.
# Original boundary guards/order preserved, including asymmetric bottom-corner condition.
@inline ewidx(j,i,NLat)=(i-Int32(1))*NLat+j-Int32(1)
function ew_stage1!(node,R6,Imin,Jmin,Imax,Jmax,NLat,Par_sshZeroThreshold,Par_sshArrivalThreshold,Par_time,j,i)
    if i <= Imax && j <= Jmax
        m = ewidx(j,i,NLat)
        if node[(m)*Int32(12)+Int32(0)+Int32(1)] == 0
            return nothing
        end
        node[(m)*Int32(12)+Int32(1)+Int32(1)] = node[(m)*Int32(12)+Int32(1)+Int32(1)] - node[(m)*Int32(12)+Int32(5)+Int32(1)]*( node[(m)*Int32(12)+Int32(3)+Int32(1)] - node[(m-NLat)*Int32(12)+Int32(3)+Int32(1)] + node[(m)*Int32(12)+Int32(4)+Int32(1)]*R6[j+Int32(1)] - node[(m-1)*Int32(12)+Int32(4)+Int32(1)]*R6[j-1+Int32(1)] )
        absH = abs(node[(m)*Int32(12)+Int32(1)+Int32(1)])
        if absH < Par_sshZeroThreshold
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = 0.
        end
        if node[(m)*Int32(12)+Int32(1)+Int32(1)] > node[(m)*Int32(12)+Int32(2)+Int32(1)]
            node[(m)*Int32(12)+Int32(2)+Int32(1)] = node[(m)*Int32(12)+Int32(1)+Int32(1)]
        end
        if (Par_sshArrivalThreshold != 0f0) && node[(m)*Int32(12)+Int32(10)+Int32(1)] < 0 && absH > Par_sshArrivalThreshold
            node[(m)*Int32(12)+Int32(10)+Int32(1)] = Float32(Par_time)
        end
    end
    nothing
end
function ew_stage2!(node,C1,C2,C3,C4,Imin,Jmin,Imax,Jmax,NLat,NLon)
    if Jmin <= 2
        for i=Int32(2):Int32((NLon-1))
            m = ewidx(1,i,NLat)
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m)*Int32(12)+Int32(4)+Int32(1)])^( 2.0f0)) + 
          0.25f0*(((node[(m)*Int32(12)+Int32(3)+Int32(1)]+node[(m-NLat)*Int32(12)+Int32(3)+Int32(1)]))^(2.0f0)) )*C1[i+Int32(1)]
            if node[(m)*Int32(12)+Int32(4)+Int32(1)] > 0
                node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
            end
        end
    end
    if Imin <= 2
        for j=Int32(2):Int32((NLat-1))
            m = ewidx(j,1,NLat)
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m)*Int32(12)+Int32(3)+Int32(1)])^(2.0f0)) + 
          0.25f0*(((node[(m)*Int32(12)+Int32(4)+Int32(1)]+node[(m-1)*Int32(12)+Int32(4)+Int32(1)]))^(2.0f0)) )*C2[j+Int32(1)]
            if node[(m)*Int32(12)+Int32(3)+Int32(1)] > 0
                node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
            end
        end
    end
    if Jmax >= (NLat-1)
        for i=Int32(2):Int32((NLon-1))
            m = ewidx(NLat,i,NLat)
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m-1)*Int32(12)+Int32(4)+Int32(1)])^(2.0f0)) + 0.25f0*(((node[(m)*Int32(12)+Int32(3)+Int32(1)]+node[(m-1)*Int32(12)+Int32(3)+Int32(1)]))^(2.0f0)) )*C3[i+Int32(1)]
            if node[(m-1)*Int32(12)+Int32(4)+Int32(1)] < 0
                node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
            end
        end
    end
    if Imax >= (NLon-1)
        for j=Int32(2):Int32((NLat-1))
            m = ewidx(j,NLon,NLat)
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m-NLat)*Int32(12)+Int32(3)+Int32(1)])^(2.0f0)) + 0.25f0*(((node[(m)*Int32(12)+Int32(4)+Int32(1)]+node[(m-1)*Int32(12)+Int32(4)+Int32(1)]))^(2.0f0)) )*C4[j+Int32(1)]
            if node[(m-NLat)*Int32(12)+Int32(3)+Int32(1)] < 0
                node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
            end
        end
    end
    if Jmin <= 2
        m = ewidx(1,1,NLat)
        node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m)*Int32(12)+Int32(3)+Int32(1)])^(2.0f0)) + ((node[(m)*Int32(12)+Int32(4)+Int32(1)])^(2.0f0)) )*C1[1+Int32(1)]
        if node[(m)*Int32(12)+Int32(4)+Int32(1)] > 0
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
        end
        m = ewidx(1,NLon,NLat)
        node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m-NLat)*Int32(12)+Int32(3)+Int32(1)])^(2.0f0)) + ((node[(m)*Int32(12)+Int32(4)+Int32(1)])^(2.0f0)) )*C1[NLon+Int32(1)]
        if node[(m)*Int32(12)+Int32(4)+Int32(1)] > 0
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
        end
    end
    if Jmin >= (NLat-1)
        m = ewidx(NLat,1,NLat)
        node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m)*Int32(12)+Int32(3)+Int32(1)])^(2.0f0)) + ((node[(m-1)*Int32(12)+Int32(4)+Int32(1)])^(2.0f0)) )*C3[1+Int32(1)]
        if node[(m-1)*Int32(12)+Int32(4)+Int32(1)] < 0
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
        end
        m = ewidx(NLat,NLon,NLat)
        node[(m)*Int32(12)+Int32(1)+Int32(1)] = sqrt( ((node[(m-NLat)*Int32(12)+Int32(3)+Int32(1)])^(2.0f0)) + ((node[(m-1)*Int32(12)+Int32(4)+Int32(1)])^(2.0f0)) )*C3[NLon+Int32(1)]
        if node[(m-1)*Int32(12)+Int32(4)+Int32(1)] < 0
            node[(m)*Int32(12)+Int32(1)+Int32(1)] = - node[(m)*Int32(12)+Int32(1)+Int32(1)]
        end
    end
    nothing
end
function ew_stage3!(node,R6,Imin,Jmin,Imax,Jmax,NLat,j,i)
    if i <= Imax && j <= Jmax
        m = ewidx(j,i,NLat)
        if (node[(m)*Int32(12)+Int32(0)+Int32(1)]*node[(m+NLat)*Int32(12)+Int32(0)+Int32(1)]) != 0
            node[(m)*Int32(12)+Int32(3)+Int32(1)] = node[(m)*Int32(12)+Int32(3)+Int32(1)] - node[(m)*Int32(12)+Int32(6)+Int32(1)]*(node[(m+NLat)*Int32(12)+Int32(1)+Int32(1)]-node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
        if (node[(m)*Int32(12)+Int32(0)+Int32(1)]*node[(m+1)*Int32(12)+Int32(0)+Int32(1)]) != 0
            node[(m)*Int32(12)+Int32(4)+Int32(1)] = node[(m)*Int32(12)+Int32(4)+Int32(1)] - node[(m)*Int32(12)+Int32(8)+Int32(1)]*(node[(m+1)*Int32(12)+Int32(1)+Int32(1)]-node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
    end
    nothing
end
function ew_stage4!(node,C1,C2,C3,C4,Imin,Jmin,Imax,Jmax,NLat,NLon,Par_sshClipThreshold)
    if Jmin[1] <= 2
        for i=Int32(1):Int32((NLon-1))
            m = ewidx(1,i,NLat)
            node[(m)*Int32(12)+Int32(3)+Int32(1)] = node[(m)*Int32(12)+Int32(3)+Int32(1)] - node[(m)*Int32(12)+Int32(6)+Int32(1)]*(node[(m+NLat)*Int32(12)+Int32(1)+Int32(1)] - node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
    end
    if Imin[1] <= 2
        for j=Int32(1):Int32(NLat)
            m = ewidx(j,1,NLat)
            node[(m)*Int32(12)+Int32(3)+Int32(1)] = node[(m)*Int32(12)+Int32(3)+Int32(1)] - node[(m)*Int32(12)+Int32(6)+Int32(1)]*(node[(m+NLat)*Int32(12)+Int32(1)+Int32(1)] - node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
    end
    if Jmax[1] >= (NLat-1)
        for i=Int32(1):Int32((NLon-1))
            m = ewidx(NLat,i,NLat)
            node[(m)*Int32(12)+Int32(3)+Int32(1)] = node[(m)*Int32(12)+Int32(3)+Int32(1)] - node[(m)*Int32(12)+Int32(6)+Int32(1)]*(node[(m+NLat)*Int32(12)+Int32(1)+Int32(1)] - node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
    end
    if Imin[1] <= 2
        for j=Int32(1):Int32((NLat-1))
            m = ewidx(j,1,NLat)
            node[(m)*Int32(12)+Int32(4)+Int32(1)] = node[(m)*Int32(12)+Int32(4)+Int32(1)] - node[(m)*Int32(12)+Int32(8)+Int32(1)]*(node[(m+1)*Int32(12)+Int32(1)+Int32(1)] - node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
    end
    if Jmin[1] <= 2
        for i=Int32(1):Int32(NLon)
            m = ewidx(1,i,NLat)
            node[(m)*Int32(12)+Int32(4)+Int32(1)] = node[(m)*Int32(12)+Int32(4)+Int32(1)] - node[(m)*Int32(12)+Int32(8)+Int32(1)]*(node[(m+1)*Int32(12)+Int32(1)+Int32(1)] - node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
    end
    if Imax[1] >= (NLon-1)
        for j=Int32(1):Int32((NLat-1))
            m = ewidx(j,NLon,NLat)
            node[(m)*Int32(12)+Int32(4)+Int32(1)] = node[(m)*Int32(12)+Int32(4)+Int32(1)] - node[(m)*Int32(12)+Int32(8)+Int32(1)]*(node[(m+1)*Int32(12)+Int32(1)+Int32(1)] - node[(m)*Int32(12)+Int32(1)+Int32(1)])
        end
    end
    if Imin[1] > 2
        enlarge=0
        for j=Int32(Jmin[1]):Int32(Jmax[1])
            if abs(node[(ewidx(j,Imin[1]+2,NLat))*Int32(12)+Int32(1)+Int32(1)]) > Par_sshClipThreshold
                enlarge = 1
                break
            end
        end
        if enlarge != 0
            Imin[1] -= Int32(1)
            if Imin[1] < 2
                Imin[1] = 2
            end
        end
    end
    if Imax[1] < (NLon-1)
        enlarge=0
        for j=Int32(Jmin[1]):Int32(Jmax[1])
            if abs(node[(ewidx(j,Imax[1]-2,NLat))*Int32(12)+Int32(1)+Int32(1)]) > Par_sshClipThreshold
                enlarge = 1
                break
            end
        end
        if enlarge != 0
            Imax[1] += Int32(1)
            if Imax[1] > (NLon-1)
                Imax[1] = NLon-1
            end
        end
    end
    if Jmin[1] > 2
        enlarge=0
        for i=Int32(Imin[1]):Int32(Imax[1])
            if abs(node[(ewidx(Jmin[1]+2,i,NLat))*Int32(12)+Int32(1)+Int32(1)]) > Par_sshClipThreshold
                enlarge = 1
                break
            end
        end
        if enlarge != 0
            Jmin[1] -= Int32(1)
            if Jmin[1] < 2
                Jmin[1] = 2
            end
        end
    end
    if Jmax[1] < (NLat-1)
        enlarge=0
        for i=Int32(Imin[1]):Int32(Imax[1])
            if abs(node[(ewidx(Jmax[1]-2,i,NLat))*Int32(12)+Int32(1)+Int32(1)]) > Par_sshClipThreshold
                enlarge = 1
                break
            end
        end
        if enlarge != 0
            Jmax[1] += Int32(1)
            if Jmax[1] > (NLat-1)
                Jmax[1] = NLat-1
            end
        end
    end
    nothing
end
