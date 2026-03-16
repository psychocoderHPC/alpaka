/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Assert.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/meta/Integral.hpp"
#include "aV2P/meta/NdLoop.hpp"

#include <cstring>

namespace aV2P
{
    class DevCpu;

    namespace detail
    {
        template<typename TView, typename TExtents>
        struct TaskSetAlpaka3
        {
            TaskSetAlpaka3(TView&& viewDst, uint8_t byte, TExtents const& extent)
                : m_dest{std::forward<TView>(viewDst)}
                , m_extents{extent}
                , m_value{byte}
            {
            }

            TView m_dest;
            TExtents m_extents;
            uint8_t m_value;
        };
    } // namespace detail

    namespace trait
    {
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TDim>
        struct CreateTaskMemset<TDim, DevAlpaka3<TDevSpec>>
        {
            template<typename TExtents, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(
                TView&& viewDst,
                std::uint8_t const& byte,
                TExtents const& extent)
            {
                return aV2P::detail::TaskSetAlpaka3{viewDst.getAlpaka3().getView(), byte, extent.getAlpaka3()};
            }
        };

        template<typename TDev, typename TView, typename TExtents>
        struct Enqueue<QueueAlpaka3CpuNonBlocking<TDev>, aV2P::detail::TaskSetAlpaka3<TView, TExtents>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3CpuNonBlocking<TDev>& queue,
                aV2P::detail::TaskSetAlpaka3<TView, TExtents> task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                alpaka::onHost::memset(queue.getAlpaka3(), task.m_dest, task.m_value, task.m_extents);
            }
        };

        template<typename TDev, typename TView, typename TExtents>
        struct Enqueue<QueueAlpaka3CpuBlocking<TDev>, aV2P::detail::TaskSetAlpaka3<TView, TExtents>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3CpuBlocking<TDev>& queue,
                aV2P::detail::TaskSetAlpaka3<TView, TExtents> task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                alpaka::onHost::memset(queue.getAlpaka3(), task.m_dest, task.m_value, task.m_extents);
            }
        };
    } // namespace trait
} // namespace aV2P
