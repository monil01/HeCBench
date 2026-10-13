Retains all four original LDPC kernels, encoder, modulation/noise stream,
seed69012, matrix data and error statistics. Eight buffers use dmem. LLR
resets with update for each simulation; final hard decisions flush before
error_check. The CNP dynamic workspace has compile-time-known dimensions
and becomes an identically indexed static shared array of that exact size.
Defaults MCW40/MAX_SIM500/MIN_CODEWORD2000/MAX_ITERATION10 are preserved.
VALIDATION=yes bounds only batch/simulation counts to2/2/0. Isolated native
copy uses identical counts and adds complete hard-decision dumps. Every
decoded bit and the original BER/FER statistics are compared exactly.
