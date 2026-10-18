# CHE predeclared periodic conserved-phase study

{
  "cases": [
    [
      16,
      8,
      8,
      1,
      0
    ],
    [
      19,
      9,
      7,
      3,
      0
    ],
    [
      16,
      8,
      8,
      3,
      1
    ],
    [
      16,
      8,
      8,
      3,
      2
    ]
  ],
  "precision": "Float64",
  "parameters": {
    "dx_dy_dz": 1.0,
    "dt": 0.01,
    "gamma": 0.5,
    "D": 1.0,
    "eAA_eBB": -0.2222222222222222,
    "eAB": 0.2222222222222222
  },
  "fixture": "Sharedserializedsigneddyadic,constant0.25,singlexFouriermode amplitude1/16",
  "outputs": "All4rawfields current/previousc,mu,localfreeenergy and32exactguards; sourceperiodicwrap allfaces",
  "geometry": [
    16,
    4,
    4
  ],
  "timing": "Sum4preallocatedsource4launchstep pipelines withcontextcompletion; excludesoriginal diagnosticD2H/integrals/fileI/O consistentlyall4",
  "scope": "Namedsmallgridsand1/3steps, includingpartialtiles; no256cubed/25000stepCLI orcoarseningvalidation",
  "mass_conservation_absolute_bound": 1e-10
}

Finiteallfields,1e-11+1e-11rel independent anddualreferenceacceptance; exactinputprefixandallallocationguards; absoluteperiodicmassdrift<=1e-10. Constantstateanalyticfields andsingleFourierinput included. Independent localpotential identity uses exactcoefficient mathematics, allowing roundoff from sourcebinary2/9 coefficients. No energy-decay theorem or continuous PDE convergence claimed. Native sourcebodyspan retained verbatim; namespaces specialize onlydimensions.
