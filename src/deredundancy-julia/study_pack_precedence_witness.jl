# CPU-only Julia operator-precedence witness; no CUDA import or GPU work.
b=UInt32(2);r=Int32(0)
wrong=b<<(Int32(15)-r)*Int32(2)
correct=b<<((Int32(15)-r)*Int32(2))
println("wrong=",wrong," corrected=",correct)
Meta.show_sexpr(stdout,Meta.parse("b << (Int32(15)-r)*Int32(2)"));println()
@assert wrong==UInt32(131072)&&correct==UInt32(2147483648)
