# Private refresh persistence and borrowed-lock companion

UNAPPLIED, UNEXECUTED. No candidate imports, tests, rendering, builds, GPU work, process signals or supervisor changes. Root review/actual authority and supervisor readoption remain required. Original consumer packet and profiler presentation v2 packet are unchanged.

## Which candidate to review

- `minimal_passthrough.patch` adds only CLI --current537-display-review-sha256 and report-command forwarding against exact live refresh33952a56… . **HOLD: do not adopt this alone.** Existing refresh waits for report while holding metadata LOCK_EX; the original private loader's new LOCK_SH would deadlock.
- `complete_combined/minimal.patch` is the complete consumer+loader+refresh proposal against live bases. It includes all previous seven mapping/geometry decorations, fourteen curated aliases, source/hash gates,537-only label qualifications and the necessary borrowed-FD protocol. Candidate overlay bytes/source asset bindings unchanged.
- `refresh_traits_FD_only.patch` is the small refresh-only part of that combined proposal for focused review.
- `future_method_and_traits_FD/tools/v100_roofline_refresh.py` and `future_method_traits_FD_vs_method_v2.patch` explicitly compose both the trait flag/FD protocol and the prior unadopted profiler presentation v2 refresh changes. `future_method_traits_FD_vs_live.patch` is the full comparison against current live refresh. No new profiler methods/calibration pass/adoption is claimed.
- `complete_combined/root_review.template.json` pins the new actual loader SHA and unchanged overlay SHA. It remains PENDING_ROOT_REVIEW_NOT_APPROVAL. No approved canonical root_review.json or final receipt SHA is fabricated.

## Exact lock protocol

Refresh's existing LOCK_EX scope remains around report child and atomic report publication. The report command alone receives the explicit trait authority SHA and --metadata-lock-fd N. subprocess.run receives pass_fds=(N,) only for that report call; all other children retain an empty tuple. N is the exact currently held parent lock object's fileno; no environment/global descriptor guessing or arbitrary wildcard is used.

Report and graph accept explicit --metadata-lock-fd and pass it to the shared loader. The loader compares fstat(fd) device/inode and regular-file type to the canonical existing metadata-lock file. It requests LOCK_EX|LOCK_NB on the inherited shared open-file description: same-description EX→EX succeeds without downgrading, or a competing/different lease fails immediately rather than blocking. Borrowed context exit never calls unlock or close. Parent retains its EX lease until its unchanged enclosing context ends. No new privilege is involved. A failed descriptor check withholds decoration through the existing authority/entry failure gates.

Standalone callers without an outer EX lease omit --metadata-lock-fd and own a separate read-only LOCK_SH context, which they unlock/close normally. Root manual report/graph calls already under EX must inherit the exact held FD through pass_fds and supply --metadata-lock-fd; supplying authority while omitting FD under an outer EX would retain the documented deadlock. Both CLI signatures support this protocol.447 remains undecorated and its data/filter/scoring behavior unchanged.

## Root action after independent review

1. Adopt combined exact bytes, create actual approved root_review.json from the updated pending template only after review, calculate FINAL_ROOT_REVIEW_SHA. Candidate overlay SHA remains9b0788ff1472540bb5b22a8a5c847d63dcd9e8801a6cfb029c7e07b95424ab6c.
2. Root alone handles the exact owned idle refresh1215425/no-child checks and supervisor v13→v14 source pins. This worker sent no signals, queried no GPU, launched no report/profiler/controller child and changed no controller frontier.
3. Restart reviewed refresh with existing absolute source/raw roots, exact owned wait-PID arguments and --current537-display-review-sha256 FINAL_ROOT_REVIEW_SHA. Only report receives it; graph remains manually regenerated with explicit authority/borrowed-FD arguments as appropriate.
4. Actual final root receipt SHA is an argument variable, not an inferred/default approval. Future method presentation must adopt the composed candidate to retain this forwarding/protocol. Actual calibration002/method admission remain separate gates.
5. Root may independently run syntax/render/lock/edge-inventory checks after review. None were run here. Frozen scientific records, numerical outcomes, original timings, source assets, graph scoring fields and edge algorithms remain unchanged.
