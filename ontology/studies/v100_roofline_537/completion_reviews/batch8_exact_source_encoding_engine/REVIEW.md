# Private exact source-encoding merge extension

Base: flip_exact_merge_exception_proposal/merge_flip_exact.py SHA fad948a2a809e0cc689d115914bb6bf5ac9ecdd2c887cced84f53bfb7d624cd8. Candidate: merge_batch8_exact_encoding.py. Review source.patch before root adoption. No engine execution, import, compile, tests, dry run or shared scientific writes occurred.

The root-note field approved_reversible_source_encodings is a mapping keyed by the exact two absolute avilib paths. Entries must equal the hardcoded file SHA and encoding latin-1; additional paths, altered hashes/encoding or entry keys fail. Each encountered avilib source requires its exact root approval. The fragment JSON is a proposed note fragment only, without approval authority.

Source bytes must still match both the proof hash and selected captured binding before decoding. The existing eigenvalue exact-path/hash Latin-1 exception is retained. Full text must encode back to the original bytes. Snippets use the same approved encoding in both directions, matching the exact byte interval; any declared snippet_encoding must agree. Existing byte offset bounds and start/end line proof, exact source lines/context and source/capture hashes remain enforced. All other files use strict UTF8. No replacement, ignored errors, generic fallback or filesystem mutation is introduced.

Changes are confined to the new root-note mapping gate and source quotation decoding. All other whitelists, counts, source-family checks, parent/index joins, flip exact contract exception, raw-counter invariants and merge/application machinery are untouched. Runtime behavior is unexecuted and remains for root's separate review and authorized validation.
