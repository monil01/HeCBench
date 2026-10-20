#!/usr/bin/env python3
"""CPU-only independent analytic witness of the source's conductive subtraction."""
import math,json,pathlib,hashlib
N=1000;M=500;dx=1/(M-1)
# Ideal source mathematical input at the last three internal vertical rows.
t=[1-(M-4)*dx,1-(M-3)*dx,1-(M-2)*dx,0.]
z=[1-3*dx,1-2*dx,1-dx,1.]
perturbation=[a+b-1 for a,b in zip(t,z)]
source_derivative=sum(a*b for a,b in zip([-2/3,3,-6,11/3],perturbation))/(2*dx)
# Source trapezoid weights sum (N-1)*dx, divided by negative aspect2.
source_value=-source_derivative*(N-1)*dx/2
true_total_heat_flux=1.
result={'input':'Analytic conducting equilibrium T(z)=1-z; psi=u=v=0','source_ideal_return':source_value,'total_dimensionless_Nusselt':true_total_heat_flux,'pass':abs(source_value)<1e-10,'diagnosis':'Source returns conductive-background-subtracted heat flux, not total Nusselt. Keep actual source value as excess diagnostic; do not label it validated totalNu.','not_a_GPU_failure':'CPU exact mathematical witness of diagnostic definition; no observed originalGPU exception or solver accuracy failure inferred.','source_path':'reference_history/main.cu','source_sha256':hashlib.sha256((pathlib.Path(__file__).parent/'reference_history/main.cu').read_bytes()).hexdigest()}
assert result['pass'];print(json.dumps(result,indent=2))
