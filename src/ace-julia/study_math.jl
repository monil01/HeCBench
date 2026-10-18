module Ace16
using CUDA
const DATAXSIZE=Int32(16);const DATAYSIZE=Int32(16);const DATAZSIZE=Int32(16)
include("study_math.inc.jl")
end
module Ace24
using CUDA
const DATAXSIZE=Int32(24);const DATAYSIZE=Int32(24);const DATAZSIZE=Int32(24)
include("study_math.inc.jl")
end
