/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/CudaHipCommon.hpp"
#include "aV2P/core/UniformCudaHip.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#    if !ALPAKA_LANG_HIP && !defined(ALPAKA_HOST_ONLY)
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif
#endif
