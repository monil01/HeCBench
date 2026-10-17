// Execute unchanged original CPU reference; first nonzero byte writes past256-bytealphabet.
#include <vector>
#include <algorithm>
#include "../mtf-cuda/reference.h"
int main(){std::vector<char>w{'A'};auto got=reference(w);return got.empty();}
