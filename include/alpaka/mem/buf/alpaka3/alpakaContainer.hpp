/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace alpaka
{
    namespace internal
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct GetApi::Op<BufCpu<TElem, TDim, TIdx>>
        {
            inline constexpr auto operator()(auto&& buf) const
            {
                alpaka::unused(buf);
                return api::Host{};
            }
        };
    } // namespace internal

    namespace onHost::internal
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct GetExtents::Op<BufCpu<TElem, TDim, TIdx>>
        {
            decltype(auto) operator()(auto&& buf) const
            {
                auto ex = alpaka::getExtents(buf);
                return ex.getAlpaka3();
            }
        };

        template<typename TElem, typename TDim, typename TIdx>
        struct GetPitches::Op<BufCpu<TElem, TDim, TIdx>>
        {
            decltype(auto) operator()(auto&& buf) const
            {
                auto ex = alpaka::getPitchesInBytes(buf);
                return ex.getAlpaka3();
            }
        };
    } // namespace onHost::internal

    namespace trait
    {
        template<typename TElem, typename TDim, typename TIdx>
        struct GetValueType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };

        template<typename TElem, typename TDim, typename TIdx>
        struct GetDim<BufCpu<TElem, TDim, TIdx>>
        {
            static constexpr uint32_t value = TDim::value;
        };
    } // namespace trait
} // namespace alpaka
