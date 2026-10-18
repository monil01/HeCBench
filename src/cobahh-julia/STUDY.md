# cobahh: scientific and implementation observation

Source comments retain tiny representable nonzero rate coefficients, but actual compiled literals are zero. The original native GPU produces negative activation probability m=-0.32084724 from a physiological initial state. Independent regrouped affine-rate algebra confirms this source behavior and its violation of the positive-rate gate-probability contract. Four-way agreement is therefore a native-behavior result, explicitly excluded from faithful-scientific-operator aggregates. This is a zero-dimensional biophysical membrane ODE case within the continuum/constitutive family, not a machine-learning operator or spatial PDE.

## Reference audit

Historical CUDA.jl smoke computes no neuronal update and prints a fake zero RSME. Restore actual native coefficient/state update, not calibrated HH physics. Original native source retained; zero gate-rate coefficients cause a separately witnessed native scientific defect. An initial restored Float64-literal arithmetic candidate passed arrays but violated requested Float32 computation: retained as P1 precision alignment, then corrected and independently rechecked.

## Numerical and performance evidence

Configuration `cobahh_actual_zero_coefficients_threefixtures_33_257_129_steps1_1_3_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T041642255538Z_cobahh/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Current CUDA.jl restoration has two saved source candidates and two whole-program GPU checks: the first passed outputs but accidentally used mixed Float64 literals, corrected under P1 precision alignment. No compiler failure or numerical tolerance repair. KA and JACC first candidate/check passes. Historical smoke effort unknown; native scientific defect is independently attributed to CUDA source, not target frameworks. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective phase activity counters retained including repeated/cached context and marker lag; nonexclusive, not billed cost. Historical usage unknown and saved-source proxies partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/cobahh-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Scientific defect and complete structural evidence

All20captures contain3047exactprefix+3352complete states/flags+48guards=6447Float32values. Every serialized input/guard and readonlylastspike array is independently bitchecked; all33physiological m gates are negative in every capture. Native header is untouched, so this is known-defective native behavior, excluded from scientific-fidelity and untouched-native aggregates; no calibrated HH/neural-network claim. The original independent-oracle JSON prospectively states native GPU confirmation pending; native_gating_defect_witness.json and independent_capture_witness.json provide the later actual confirmation without altering the hash-bound original. Family continuum/constitutive, zero-dimensional membraneODE, not spatialPDE. Means seconds {"cuda": 2.8597833333333333e-05, "cuda_jl": 6.488816666666667e-05, "ka": 0.00016728483333333335, "jacc": 0.00013642533333333334}.
