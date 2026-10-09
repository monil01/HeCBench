# streamCreateCopyDestroy IRIS port

This benchmark measures HIP stream creation, asynchronous host-to-device copy,
synchronization, and stream destruction overhead. IRIS dmem does not expose
native stream create/destroy controls directly, so this port preserves the
same test matrix and host-to-device copy volume with `iris::DMem::update()`.

The printed `[Stream]` rows are therefore a portability comparison for IRIS
dmem copy submission overhead, not a one-to-one implementation of HIP stream
creation/destruction.
