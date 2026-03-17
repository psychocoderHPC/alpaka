/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan, Bernhard
 * Manfred Gruber
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
    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    class DevAlpaka3;
} // namespace aV2P

namespace aV2P
{
    namespace detail
    {
        //! The CPU device memory copy task base.
        //!
        //! Copies from CPU memory into CPU memory.
        template<typename TViewDst, typename TViewSrc, typename TExtents>
        struct TaskCopyAlpaka3Cpu
        {
            static_assert(TExtents::dim() > 0);

            TaskCopyAlpaka3Cpu(TViewDst viewDst, TViewSrc const& viewSrc, TExtents const& extent)
                : m_dest{viewDst}
                , m_src{viewSrc}
                , m_extents{extent}
            {
            }

            TViewDst m_dest;
            TViewSrc m_src;
            TExtents m_extents;
        };
    } // namespace detail

    namespace trait
    {
        //! The CPU device memory copy trait specialization.
        //!
        //! Copies from CPU memory into CPU memory.
        template<
            alpaka::onHost::concepts::DeviceSpec TDevSpecDst,
            alpaka::onHost::concepts::DeviceSpec TDevSpecSrc,
            typename TDim>
        struct CreateTaskMemcpy<TDim, DevAlpaka3<TDevSpecDst>, DevAlpaka3<TDevSpecSrc>>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                if constexpr(std::is_integral_v<TExtent>)
                    return aV2P::detail::TaskCopyAlpaka3Cpu{
                        viewDst.getAlpaka3(),
                        viewSrc.getAlpaka3(),
                        alpaka::Vec{extent}};
                else
                    return aV2P::detail::TaskCopyAlpaka3Cpu{
                        viewDst.getAlpaka3(),
                        viewSrc.getAlpaka3(),
                        extent.getAlpaka3()};
            }
        };

        template<typename TDev, typename TViewDst, typename TViewSrc, typename TExtent, typename TQueueKind>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, aV2P::detail::TaskCopyAlpaka3Cpu<TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                aV2P::detail::TaskCopyAlpaka3Cpu<TViewDst, TViewSrc, TExtent> task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                alpaka::onHost::memcpy(queue.getAlpaka3(), task.m_dest, task.m_src, task.m_extents);
            }
        };
    } // namespace trait
} // namespace aV2P
