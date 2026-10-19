struct NNCase {int n,k,repeats,offset;float lat,lng;};
static const NNCase cases[] = {
{1,1,10000,0,30.0f,90.0f},
{17,5,10000,34,-10.0f,-20.0f},
{65,5,10000,100,0.0f,0.0f},
{4099,5,10000,262,30.0f,90.0f},
{131072,5,10000,8492,30.0f,90.0f},
};
static constexpr size_t INPUT_FLOATS=270668,PREFIX_COUNT=541376,OUTPUT_COUNT=947298;
