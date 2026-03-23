/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Assert.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/meta/Integral.hpp"
#include "aV2P/meta/NdLoop.hpp"

namespace aV2P
{
    namespace detail
    {
        template<typename TView, typename Elem, typename TExtents>
        struct TaskFillAlpaka3
        {
            TaskFillAlpaka3(TView&& viewDst, Elem value, TExtents const& extent)
                : m_dest{std::forward<TView>(viewDst)}
                , m_extents{extent}
                , m_value{value}
            {
            }

            TView m_dest;
            TExtents m_extents;
            Elem m_value;
        };
    } // namespace detail

    namespace trait
    {
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TDim>
        struct CreateTaskFill<TDim, DevAlpaka3<TDevSpec>>
        {
            template<typename TExtents, typename TView>
            ALPAKA_FN_HOST static auto createTaskFill(
                TView&& viewDst,
                aV2P::Elem<std::remove_reference_t<TView>> const& value,
                TExtents const& extent)
            {
                if constexpr(std::is_integral_v<TExtents>)
                    return aV2P::detail::TaskFillAlpaka3{viewDst.getAlpaka3(), value, alpaka::Vec{extent}};
                else
                    return aV2P::detail::TaskFillAlpaka3{viewDst.getAlpaka3(), value, extent.getAlpaka3()};
            }
        };

        template<typename TDev, typename TView, typename TElem, typename TExtents, typename TQueueKind>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, aV2P::detail::TaskFillAlpaka3<TView, TElem, TExtents>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                aV2P::detail::TaskFillAlpaka3<TView, TElem, TExtents> task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                alpaka::onHost::fill(queue.getAlpaka3(), task.m_dest, task.m_value, task.m_extents);
            }
        };
    } // namespace trait
} // namespace aV2P
