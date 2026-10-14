#pragma once
#include "normalize.h"

void linearize(const IndexSpec& spec, at::Tensor& idx_out);
