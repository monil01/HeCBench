IRIS DMem port of the original Rodinia B+ tree point/range kernels.
Default data/command paths and CLI are preserved. Original DVC archive
src/data/b+tree/b+tree.tar.bz.dvc: MD5 34c0cb9e186394582ef8de92219d2530,
size1211592, with mil.txt and command.txt. Data is external and not committed.
Current validation recovered this public checksummed archive into local
porting state, built current native HIP and ran the full million-key input,
10000point and6000range queries on actual HIP with exact artifact/reference
verification. Original command parser treats j60003000 as range6000;
that native behavior is preserved.
Output DMem uses iris_rw to preserve native initialized sentinels for
unwritten entries. GPU lock is held within wrappers; CPU tree construction
and external verification run outside. Earlier attempts remain intact.
