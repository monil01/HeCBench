Status: unverified; whole-output native comparison failed on native NaNs.

Pitch retains all11shape defaults and1000repetitions. HECBENCH_DUMP optionally
writes every logical output for pitched/simple2D/3D; padding is excluded.
HECBENCH_PITCH_SHAPES optionally bounds validation to the first1..11original
shapes. Explicit128-byte padded row strides are represented by IRIS dmem.
Native allocation contents are uninitialized; validation preserves that
baseline and checks final1000-repeat outputs. IRIS retains its existing zero
host initialization. A passing comparison demonstrates the tested final
outputs, not equivalence of unspecified initial memory or every repeat count.

The dump-only native adapter exposed NaNs that persist through sigmoid from
uninitialized native allocation contents. IRIS's pre-existing zero input
does not reproduce those NaNs. No native initialization repair was applied;
full numerical verification remains blocked pending baseline scope approval.
