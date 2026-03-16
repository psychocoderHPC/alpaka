/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"
#include "aV2P/dev/DevAlpaka3Cpu.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/event/Traits.hpp"
#include "aV2P/queue/Traits.hpp"
#include "aV2P/wait/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <type_traits>

namespace aV2P
{
    template<typename TDev>
    class EventAlpaka3Cpu;

    //! The CPU device queue.
    template<typename TDev>
    class QueueAlpaka3CpuBlocking final
        : public interface::Implements<ConceptCurrentThreadWaitFor, QueueAlpaka3CpuBlocking<TDev>>
        , public interface::Implements<ConceptQueue, QueueAlpaka3CpuBlocking<TDev>>
        , public interface::Implements<ConceptGetDev, QueueAlpaka3CpuBlocking<TDev>>
    {
    public:
        using QueueType = alpaka::onHost::Queue<typename TDev::DevType, alpaka::queueKind::Blocking>;

        auto getAlpaka3() const
        {
            return m_queueImpl;
        }

        explicit QueueAlpaka3CpuBlocking(TDev const& dev)
            : m_queueImpl{dev.getAlpaka3().makeQueue(alpaka::queueKind::blocking)}
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;
        }

        auto operator==(QueueAlpaka3CpuBlocking<TDev> const& rhs) const -> bool
        {
            return (m_queueImpl == rhs.m_queueImpl);
        }

        auto operator!=(QueueAlpaka3CpuBlocking<TDev> const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

    public:
        QueueType m_queueImpl;
    };

    using QueueCpuBlocking = QueueAlpaka3CpuBlocking<DevCpu>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using QueueCudaRtBlocking = QueueAlpaka3CpuBlocking<DevCudaRt>;
#endif

    namespace trait
    {
        //! The CPU non-blocking device queue device type trait specialization.
        template<typename TDev>
        struct DevType<QueueAlpaka3CpuBlocking<TDev>>
        {
            using type = TDev;
        };

        //! The CPU non-blocking device queue device get trait specialization.
        template<typename TDev>
        struct GetDev<QueueAlpaka3CpuBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto getDev(QueueAlpaka3CpuBlocking<TDev> const& queue) -> TDev
            {
                return queue.getAlpaka3().m_dev;
            }
        };

        //! The CPU non-blocking device queue event type trait specialization.
        template<typename TDev>
        struct EventType<QueueAlpaka3CpuBlocking<TDev>>
        {
            using type = EventAlpaka3Cpu<TDev>;
        };

        //! The CPU non-blocking device queue enqueue trait specialization.
        //! This default implementation for all tasks directly invokes the function call operator of the task.
        template<typename TDev, typename TTask>
        struct Enqueue<QueueAlpaka3CpuBlocking<TDev>, TTask>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueAlpaka3CpuBlocking<TDev>& queue, TTask const& task) -> void
            {
                queue.getAlpaka3().enqueueHostFn(task);
            }
        };

        //! The CPU non-blocking device queue test trait specialization.
        template<typename TDev>
        struct Empty<QueueAlpaka3CpuBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto empty(QueueAlpaka3CpuBlocking<TDev> const& queue) -> bool
            {
                static_assert(sizeof(TDev) && false, "Empty<Queue> Not implemented");
                return true;
            }
        };

        //! The CPU blocking device queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<typename TDev>
        struct CurrentThreadWaitFor<QueueAlpaka3CpuBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueAlpaka3CpuBlocking<TDev> const& queue) -> void
            {
                auto q = queue.getAlpaka3();
                alpaka::onHost::wait(q);
            }
        };
    } // namespace trait
} // namespace aV2P
