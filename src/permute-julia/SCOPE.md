# QKV payload permutation

Independent NumPy5Dreshape+axistranspose input(B,T,3,NH,D)->(3,B,NH,T,D), inversebijection exact. No native/Juliaquotientremainder indexing used. AllFloat32payloadbits checked asUInt32, including NaNpayloads/signedzero/inf/subnormalcontrols; bitpreservation notNaNnumericequality.

OriginalT1024/C768/NH12 withB1 finiteuniqueFloat32words in[-1,1]; oddB2/T7/C45/NH5 finiteandIEEEpayloadcontrol plusdegenerateB1/T1/C1/NH1. MaterializedCPUinput, notoriginalRNGidentity/defaultB8.

Sum24 original permutation launch+contextsync hostwindows, R10 eachcase/blocksize; synchronizeeveryrepeat asnativepermute API. Setup/hostviews/JIT/reset/copy/oracle/captureoutside. Fivecheckedtrials mean2-5.

ExactQKVlayoutbitrearrangement only; no attentionoutput/transformermodelaccuracy, arithmeticNaNpropagationororiginalB8default performance claim.
