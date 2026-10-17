# aes: scientific and implementation observation

The historicalone-roundsubstitute is replaced by the realten-roundAES128algorithm andnativefour-threadcooperation. IndependentOpenSSL computes everycipherblock after explicitnative-state transpose; astandardknown-answervector checks the firstblock. Both encryptionanddecryption bytearrays areverified, beyond self-roundtrip. This namedpost-grayscale fixture avoids missingbitmapdependency without timinghostpreprocessing; it doesnotcertifysecurity.

## Reference audit

HistoricalJuliaexplicitlysimplifiesAES tosingleSBox/XORround androundtrip; omits native10rounds,ShiftRows,MixColumns,expandedkeys and4threadsharedstate. GenuineAES128reference nowrestoresnativealgorithm andmapping withindependentOpenSSL/NISTfullbytewitness.

## Numerical and performance evidence

Configuration `aes128_w64_h64_allbytes_encrypt_decrypt_openssl_kat_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T181147064307Z_aes/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/aes-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


Independent evidence: all20model/trial artifacts match every OpenSSL ciphertext byte and every recovered plaintext byte. The first block also matches FIPS197(2001)AppendixC.1 after explicit native-state transpose. The CPU-only `generate_fixture.py` reproduces the exact named fixture without editing kernels; OpenSSL version, key/state layout and SHA256 provenance are recorded. This does not establish arbitrary-key coverage or security certification. Prospective workflow usage began before source audit; the parent workflow interval contains the shared reference, KA, JACC and annotation intervals, so parent and children must not be summed.

Known-answer source: [NISTFIPS197,AppendixC.1](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf); [currentstandard,editorialupdatewithnoalgorithmchange](https://csrc.nist.gov/pubs/fips/197/final).
