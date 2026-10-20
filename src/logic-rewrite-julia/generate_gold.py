#!/usr/bin/env python3
import pathlib
from aiger_controls import evaluate
root=pathlib.Path(__file__).resolve().parent
lines=['#pragma once','#include <cstdint>']
for n in (4,6,9,12):
    m,t=evaluate(root/'controls'/f'control_pi{n}.aig')
    # Explicit each-output bitmask, zero-pad unused final lanes in short tables.
    per=(1<<n)+7;per//=8
    vals=[]
    for k in range(m['outputs']):
        b=t[k*per:(k+1)*per]
        vals.extend(int.from_bytes(b[j:j+8],'little') for j in range(0,len(b),8))
    lines.append(f'static const uint64_t gold_pi{n}[]={{'+','.join(str(v)+'ULL' for v in vals)+'};')
(root/'independent_truth_gold.h').write_text('\n'.join(lines)+'\n')
