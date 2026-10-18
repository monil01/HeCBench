# ACE predeclared scoped study

{
  "cases": [
    [
      16,
      1,
      0
    ],
    [
      16,
      3,
      0
    ],
    [
      24,
      3,
      1
    ],
    [
      16,
      3,
      2
    ]
  ],
  "precision": "Float64",
  "parameters": {
    "dx_dy_dz": 0.4,
    "dt": 0.01,
    "delta": 0.8,
    "epsilon": 0.07,
    "W0": 1.0,
    "D": 2.0,
    "lambda": 1.7677669529663687,
    "tau0": 0.565685424949238
  },
  "fixture": "Shared serialized exactFloat64 sphere/sourceexp, varieddyadic, constantnegativeequilibrium",
  "outputs": "All7workarrays including final/previousphi/u and finalstepforces; exactguardsonall28allocations eachtrial",
  "steps": "Actualupdates1/3 correspond original num_steps0/2 because whilet<=num_steps",
  "timing": "Sum4 preallocated original7launch step pipelines with matched context completion, reset/inputreadback outside",
  "scope": "Named16cubed/24cubed dimensions, divisible8x8x4; original400cubed CLI notmeasured; no fulltimeconvergence claim"
}

Acceptance fixed before GPU: finite all7fields, absolute+relative1e-10, dualGPUcomparisons1e-10; exact serializedinputbytes and allocationguardbytes. Independent Python algebraic discrete oracle reexpresses anisotropic gradient derivatives; equilibrium phi=-1/u=-0.8 and zero forces analytic. Full400³/timeconvergence/scientificsolidificationpredictivity notcertified. Kernelbody span native untouched; compiletime dimensions specialized. CUDAjl historical kernel mathematics genuine; study adds measurements/guards and aligns swap to native3D mapping (P1).
