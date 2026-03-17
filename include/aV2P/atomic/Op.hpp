/* Copyright 2020 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Config.hpp"

#include <algorithm>
#include <type_traits>

namespace aV2P
{
    using alpaka::onAcc::AtomicAdd;
    using alpaka::onAcc::AtomicAnd;
    using alpaka::onAcc::AtomicCas;
    using alpaka::onAcc::AtomicDec;
    using alpaka::onAcc::AtomicExch;
    using alpaka::onAcc::AtomicInc;
    using alpaka::onAcc::AtomicMax;
    using alpaka::onAcc::AtomicMin;
    using alpaka::onAcc::AtomicOr;
    using alpaka::onAcc::AtomicSub;
    using alpaka::onAcc::AtomicXor;
} // namespace aV2P
