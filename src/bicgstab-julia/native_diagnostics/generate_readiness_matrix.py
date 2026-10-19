#!/usr/bin/env python3
"""Deterministic named 32x32 five-point SPD matrix; no external TEM dataset."""
import pathlib,sys
side=32
entries=[]
for y in range(side):
 for x in range(side):
  i=y*side+x
  if y: entries.append((i+1,i-side+1,-1.0))
  if x: entries.append((i+1,i,-1.0))
  entries.append((i+1,i+1,4.25))
out=pathlib.Path(sys.argv[1])
out.write_text("%%MatrixMarket matrix coordinate real symmetric\n% named readiness fixture; 32x32 Dirichlet grid plus diagonal shift 0.25\n"+f"{side*side} {side*side} {len(entries)}\n"+"".join(f"{i} {j} {v:.17g}\n" for i,j,v in entries))
