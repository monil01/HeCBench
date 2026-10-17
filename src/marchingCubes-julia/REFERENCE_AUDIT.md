# Source contract and independent topology audit

The archived Julia program was a touch kernel with fixed native counts and PASS. It did not translate the five-stage pipeline. The current reference restores all five GPU stages and the source's actual field: y is ignored and z is used twice. Native source is unchanged.

The original timer covers generatingTriangles only. This study preserves N=1024, iso=Float32(-0.9), all coarse/fine preprocessing and compaction, counts [8296,240380,4856560,6101640], table bytes, 128-lane generation, and original observable partial-coordinate sums. No mesh arrays are produced in the original or ports. The global triangle-index checksum depends on inter-block atomic offsets and is retained as a diagnostic rather than a deterministic mesh oracle.

Independent Float64 CPU min/max scans and table-based cube enumeration reproduce all four native counts. Each GPU trial checks the full Lv1 min/max array, full sorted compacted ID lists and all 3,733,200 Lv2 min/max entries. Coordinate summaries are divided by the vertex count before a fixed 1e-3 absolute property tolerance; this is an explicit validation adapter, not a changed timed kernel. Integer ID/count errors of one are not accepted under that tolerance.

## Source-derived topology counterexample

The source uses bit15 for X-edge coordinates and bit13 for Z-edge coordinates, but tx==3 clears bit13 and tz==7 clears bit15. An independent CPU interpretation of its integer/table operations (NumPy used for arrays and the Float64 field) finds 995 local triangle-index references outside [0,local_vertex_count) per repeated y slice. First witness: tile x=98,z=1,y=0, lane98, cube case102, triangle edge4, referenced local index36 when the block has36 vertices. Full witness values and local-count/checksum enumeration are preserved in study_fixture_provenance.json. This is a source-derived logical connectivity witness, not an observed GPU memory-access failure: the source never writes an indexed mesh array. Unchanged-native observable equivalence does not certify a scientifically valid mesh. These results must be excluded from valid-mesh/scientific-fidelity aggregates.

## Input-prefix binding

The 4401 Float64 prefix values serialize iso and the actual distinctEdges/triangle/edge-position tables. The generator parses the unchanged header. Native copies those header arrays to device; Julia copies generated files whose exact bytes match the parsed arrays. The canonical prefix is loaded outside timing; source/header/fixture SHA256 hashes and generation checks bind it to actual kernel inputs. Prefix bytes are compared exactly separately from output tolerances.

## Port implementation details

The CUDA.jl reference uses flattened physical blocks while preserving the original linear lane order and reconstructed 3D indices, warp scans and reductions. A single 1412-byte arena is reinterpreted into native UInt16 vertex indices, Float32 field values and UInt32 scan sums. Unconsumed uninitialized scratch lanes are explicitly neutralized; the all-far corner, read only by the no-owner source lane and unused in outputs, is skipped. Native retains its unchanged physical multidimensional blocks and scratch allocation. Current reference compiler repairs and shared work are costed separately from target wrappers.
