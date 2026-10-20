# CPU-only polynomial checks of source-shaped boundary derivatives.
using LinearAlgebra,Printf
N=1000;R=498;dx=Float64(1f0/499f0)
z=collect(1:R).*dx;f=repeat(reshape(1 .-z,1,:),N,1)
# Independent scalar loops for conducting profile: horizontal derivatives zero,
# vertical first derivative -1 including boundary1/0 and second derivative0.
max_first=0.;max_second=0.
for j=1:R,i=1:N
 lo=j==1 ? 1. : f[i,j-1];hi=j==R ? 0. : f[i,j+1]
 global max_first=max(max_first,abs((hi-lo)/(2dx)+1))
 global max_second=max(max_second,abs((lo-2f[i,j]+hi)/(dx*dx)))
end
# DX32 implies outer wall coordinate R+1 timesDX32 is not exactly1;
# isolate this known rounded-grid boundary discrepancy rather than demandingzero.
wall_coordinate_error=abs((R+1)*dx-1)
expected_first_bound=wall_coordinate_error/(2dx)+1e-10
expected_second_bound=wall_coordinate_error/(dx*dx)+1e-8
@assert max_first<=expected_first_bound && max_second<=expected_second_bound
@printf("{\"pass\":true,\"conduction_first_derivative_max_error\":%.17g,\"second_derivative_max_error\":%.17g,\"rounded_grid_terminal_coordinate_error\":%.17g,\"boundary_rounding_explanation\":\"Float32DX*499 differs from1; retain source boundary1/0 and quantify discrepancy\",\"no_GPU_or_fullPDE_claim\":true}\n",max_first,max_second,wall_coordinate_error)
