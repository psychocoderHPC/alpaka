/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/aV2P.hpp"

#include <cstddef>

namespace aV2P::test
{
    template<typename TDim, typename TVal>
    inline constexpr auto extentBuf = []
    {
        aV2P::Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 11 - i;
        return v;
    }();

    template<typename TDim, typename TVal>
    inline constexpr auto extentSubView = []
    {
        aV2P::Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 8 - i * 2;
        return v;
    }();

    template<typename TDim, typename TVal>
    inline constexpr auto offset = []
    {
        aV2P::Vec<TDim, TVal> v;
        if constexpr(TDim::value > 0)
            for(TVal i = 0; i < TVal{TDim::value}; i++)
                v[i] = 2 + i;
        return v;
    }();
} // namespace aV2P::test
