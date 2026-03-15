/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "aV2P/mem/buf/cpu/BufCpu.hpp"

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace alpaka
{
    namespace internal
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct GetApi::Op<aV2P::BufCpu<TElem, TDim, TIdx>>
        {
            inline constexpr auto operator()(auto&& buf) const
            {
                alpaka::unused(buf);
                return alpaka::api::Host{};
            }
        };
    } // namespace internal

    namespace onHost::internal
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct GetExtents::Op<aV2P::BufCpu<TElem, TDim, TIdx>>
        {
            decltype(auto) operator()(auto&& buf) const
            {
                auto ex = aV2P::getExtents(buf);
                return ex.getAlpaka3();
            }
        };

        template<typename TElem, typename TDim, typename TIdx>
        struct GetPitches::Op<aV2P::BufCpu<TElem, TDim, TIdx>>
        {
            decltype(auto) operator()(auto&& buf) const
            {
                auto ex = aV2P::getPitchesInBytes(buf);
                return ex.getAlpaka3();
            }
        };
    } // namespace onHost::internal

    namespace trait
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct GetValueType<aV2P::BufCpu<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };

        template<typename TElem, typename TDim, typename TIdx>
        struct GetDim<aV2P::BufCpu<TElem, TDim, TIdx>>
        {
            static constexpr uint32_t value = TDim::value;
        };
    } // namespace trait
} // namespace alpaka
