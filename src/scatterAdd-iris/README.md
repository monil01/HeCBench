# scatterAdd IRIS port status

This IRIS port preserves the scatter-add semantics and passes its CPU reference
check for both default run cases. It is marked unverified in the local porting
manifest because the native HIP baseline currently reports `FAIL` with HIP
unspecified launch failures for the same default cases, so native-HIP validation
does not provide a passing reference verdict.
