# IRIS sparse-to-dense conversion

The complete vendor CSR-to-dense conversion is replaced by an IRIS DMem HIP
row-scatter kernel into an initially zero dense array. The original matrix
generator, CSR construction, CLI, defaults and repeated same-input conversion
remain unchanged. Every dense element, including structural zeros, and the
nonzero count are compared with the source input. Performance differs from
the vendor implementation. Validate with `IRIS_ARCHS=HIP make run` or override
the original arguments with `RUN_ARGS='17 19 80 3'`.
