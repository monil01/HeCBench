# Separable convolution named study

Unchanged native main/conv.cu/conv_gold.cu build with discovered sm70. Preserved existing genuine CUDA.jl kernels compute scalar17-tap rows/columns with256threads andonepixel/lane. KA/JACC translate that mapping; native instead shared-tiles8outputs/lane (rows16x4/2560Bshared,columns16x8/5184Bshared). This source-established adaptation is a performance confound, not a framework capability failure.

Same asymmetric signed dyadicfilter17coefficients andnontrivial512x256image inallfour. Zero padding andcoefficient reversal checked atalledges. IndependentFloat64CPU rowpass,Float32intermediate,Float64columnpass checks every262144row+finalcomponent abs1e-5; alldualcomparisons exact. OriginalrelativeL2criterion explicitlychecked andzero. Dyadicexactness doesnotcertify allfloatinginputs. Dimensions respect nativealignment; full8192x8192source default andpartialtiles unmeasured. Workingimages~1.5MiBfitL2, noDRAMbandwidthclaim.

Fivewarm inprocess trials,meanlastfour,10twokernelrepeats; reset/CPUcheck/transfers/artifacts outsideclock. Eachportable onecandidate/check,no repairs. JACCshmem0 explicit. Exactbilledtokensunavailable; sourcecharacter/4 proxies partial. Scientificintent imagefiltering, no specific scientific modality/data provenance.
