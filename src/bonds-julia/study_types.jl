using StaticArrays
struct BDate
 month::Int32
 day::Int32
 year::Int32
 serial::Int32
end
struct BRate
 rate::Float32
 freq::Float32
 comp::Int32
 dc::Int32
end
struct BYield
 forward::Float32
 compounding::Float32
 frequency::Float32
 intRate::BRate
 refDate::BDate
 calDate::BDate
 dc::Int32
end
struct BBond
 start::BDate
 maturity::BDate
 rate::Float32
end
struct BCoupon
 pay::BDate
 start::BDate
 finish::BDate
 amount::Float32
end
