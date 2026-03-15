/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/Tag.hpp"
#include "aV2P/acc/Traits.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/idx/Accessors.hpp"
#include "aV2P/idx/Traits.hpp"

#include <type_traits>

namespace aV2P
{

    /* oncePerGrid
     *
     * `oncePerGrid(acc)` returns true for a single thread within the kernel execution grid.
     *
     * Usually the condition is true for block 0 and thread 0, but these indices should not be relied upon.
     */

    template<concepts::Acc TAcc>
    ALPAKA_FN_ACC inline constexpr bool oncePerGrid(TAcc const& acc)
    {
        using Dim = aV2P::Dim<TAcc>;
        using Idx = aV2P::Idx<TAcc>;
        using Vec = aV2P::Vec<Dim, Idx>;

        return getIdx<Grid, Threads>(acc) == Vec::zeros();
    }

    /* oncePerBlock
     *
     * `oncePerBlock(acc)` returns true for a single thread within the block.
     *
     * Usually the condition is true for thread 0, but this index should not be relied upon.
     */

    template<concepts::Acc TAcc>
    ALPAKA_FN_ACC inline constexpr bool oncePerBlock(TAcc const& acc)
    {
        return getIdx<Block, Threads>(acc) == Vec<Dim<TAcc>, Idx<TAcc>>::zeros();
    }

} // namespace aV2P
