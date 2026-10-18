@inline function contract_lane!(tid,tensor,adj,value,output_size,N,C)
 if tid<output_size
 B=N*C;A=N*B;Y=C*Int32(18)
 f=rem(rem(tid,Y),C);Case=div(rem(tid,Y),C)+Int32(1);y=rem(div(tid,Y),N);x=div(div(tid,Y),N)
 sum=zero(eltype(value))
 @inbounds begin
if Case == 1
a = x
b = y
for d in Int32(0):N-Int32(1)
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
end
if Case == 2
a = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
for b in Int32(0):N-Int32(1)
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
end
if Case == 3
b = x
c = y
for d in Int32(0):N-Int32(1)
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
for a in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
end
if Case == 4
b = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
for a in Int32(0):N-Int32(1)
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
end
if Case == 5
d = x
e = y
adj_value = adj[d * N + e+1]
if adj_value > 0
for a in Int32(0):N-Int32(1)
for b in Int32(0):N-Int32(1)
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
end
if Case == 6
a = x
b = y
for d in Int32(0):N-Int32(1)
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
c = d
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
if Case == 7
a = x
b = y
for d in Int32(0):N-Int32(1)
e = d
adj_value = adj[d * N + e+1]
if adj_value > 0
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 8
a = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
for b in Int32(0):N-Int32(1)
c = b
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 9
a = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
b = e
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 10
b = x
c = y
for d in Int32(0):N-Int32(1)
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
a = d
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 11
b = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
for a in Int32(0):N-Int32(1)
c = a
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 12
b = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
a = e
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 13
b = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
c = e
for a in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 14
d = x
e = y
adj_value = adj[d * N + e+1]
if adj_value > 0
for a in Int32(0):N-Int32(1)
b = a
for c in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 15
d = x
e = y
adj_value = adj[d * N + e+1]
if adj_value > 0
for b in Int32(0):N-Int32(1)
c = b
for a in Int32(0):N-Int32(1)
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
end
if Case == 16
a = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
b = e
c = e
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
if Case == 17
b = x
d = y
for e in Int32(0):N-Int32(1)
adj_value = adj[d * N + e+1]
if adj_value > 0
a = e
c = e
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
if Case == 18
d = x
e = y
adj_value = adj[d * N + e+1]
if adj_value > 0
for a in Int32(0):N-Int32(1)
b = a
c = a
sum += tensor[a * A + b * B + c * C + f+1] * adj_value
end
end
end
 value[tid+1]=sum
 end
 end
 nothing
end
