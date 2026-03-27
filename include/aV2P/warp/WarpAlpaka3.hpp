/* Copyright 2022 Sergei Bastrakov, David M. Rogers, Bernhard Manfred Gruber, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccAlpaka3Cpu.hpp"
#include "aV2P/warp/Traits.hpp"

#include <cstdint>

namespace aV2P::warp
{
    namespace trait
    {
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct GetSize<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto getSize(TAcc const&) -> std::int32_t
            {
                return TAcc::getWarpSize();
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct GetSizeCompileTime<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto getSizeCompileTime(TAcc const&) -> std::int32_t
            {
                return TAcc::getWarpSize();
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct GetSizeUpperLimit<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto getSizeUpperLimit(TAcc const&) -> std::int32_t
            {
                return TAcc::getWarpSize();
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct Activemask<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto activemask(TAcc const& acc)
            {
                return alpaka::onAcc::warp::activemask(acc);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct All<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto all(TAcc const& acc, std::int32_t predicate) -> bool
            {
                return alpaka::onAcc::warp::all(acc, predicate);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct Any<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto any(TAcc const& acc, std::int32_t predicate) -> bool
            {
                return alpaka::onAcc::warp::any(acc, predicate);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct Ballot<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto ballot(TAcc const& acc, std::int32_t predicate)
            {
                return alpaka::onAcc::warp::ballot(acc, predicate);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct Shfl<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<typename T, alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto shfl(TAcc const& acc, T val, std::int32_t srcLane, std::int32_t width)
            {
                return alpaka::onAcc::warp::shfl(acc, val, srcLane, width);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct ShflUp<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<typename T, alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto shfl_up(TAcc const& acc, T val, std::uint32_t srcLane, std::int32_t width)
            {
                return alpaka::onAcc::warp::shflUp(acc, val, srcLane, width);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct ShflDown<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<typename T, alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto shfl_down(TAcc const& acc, T val, std::uint32_t srcLane, std::int32_t width)
            {
                return alpaka::onAcc::warp::shflDown(acc, val, srcLane, width);
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct ShflXor<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<typename T, alpaka::onAcc::concepts::Acc TAcc>
            static constexpr auto shfl_xor(TAcc const& acc, T val, std::int32_t srcLane, std::int32_t width)
            {
                return alpaka::onAcc::warp::shflXor(acc, val, srcLane, width);
            }
        };
    } // namespace trait
} // namespace aV2P::warp
