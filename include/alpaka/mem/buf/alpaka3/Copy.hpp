/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan, Bernhard
 * Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/buf/alpaka3/alpakaContainer.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/meta/NdLoop.hpp"

#include <cstring>

namespace alpaka
{
    class DevAlpaka3Cpu;
} // namespace alpaka

namespace alpaka
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
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevAlpaka3Cpu, DevAlpaka3Cpu>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                return alpaka::detail::TaskCopyAlpaka3Cpu{
                    viewDst.getAlpaka3().getView(),
                    viewSrc.getAlpaka3().getView(),
                    extent.getAlpaka3()};
            }
        };

        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevAlpaka3Cpu, DevCpu>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                if constexpr(std::is_integral_v<TExtent>)
                    return alpaka::detail::TaskCopyAlpaka3Cpu{
                        viewDst.getAlpaka3().getView(),
                        makeView(viewSrc),
                        Vec{extent}};
                else
                    return alpaka::detail::TaskCopyAlpaka3Cpu{
                        viewDst.getAlpaka3().getView(),
                        makeView(viewSrc),
                        extent.getAlpaka3()};
            }
        };

        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevAlpaka3Cpu>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                if constexpr(std::is_integral_v<TExtent>)
                    return alpaka::detail::TaskCopyAlpaka3Cpu{
                        makeView(std::forward<TViewDstFwd>(viewDst)),
                        viewSrc.getAlpaka3().getView(),
                        Vec{extent}};
                else
                    return alpaka::detail::TaskCopyAlpaka3Cpu{
                        makeView(std::forward<TViewDstFwd>(viewDst)),
                        viewSrc.getAlpaka3().getView(),
                        extent.getAlpaka3()};
            }
        };

        template<typename TDev, typename TViewDst, typename TViewSrc, typename TExtent>
        struct Enqueue<
            QueueAlpaka3CpuNonBlocking<TDev>,
            alpaka::detail::TaskCopyAlpaka3Cpu<TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3CpuNonBlocking<TDev>& queue,
                alpaka::detail::TaskCopyAlpaka3Cpu<TViewDst, TViewSrc, TExtent> task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                onHost::memcpy(queue.getAlpaka3(), task.m_dest, task.m_src, task.m_extents);
            }
        };

        template<typename TDev, typename TViewDst, typename TViewSrc, typename TExtent>
        struct Enqueue<QueueAlpaka3CpuBlocking<TDev>, alpaka::detail::TaskCopyAlpaka3Cpu<TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3CpuBlocking<TDev>& queue,
                alpaka::detail::TaskCopyAlpaka3Cpu<TViewDst, TViewSrc, TExtent> task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                onHost::memcpy(queue.getAlpaka3(), task.m_dest, task.m_src, task.m_extents);
            }
        };
    } // namespace trait
} // namespace alpaka
