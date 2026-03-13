/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Config.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/event/Traits.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <type_traits>

namespace alpaka
{
    template<typename TDev>
    class EventAlpaka3Cpu;

    //! The CPU device queue.
    template<typename TDev>
    class QueueAlpaka3CpuNonBlocking final
        : public interface::Implements<ConceptCurrentThreadWaitFor, QueueAlpaka3CpuNonBlocking<TDev>>
        , public interface::Implements<ConceptQueue, QueueAlpaka3CpuNonBlocking<TDev>>
        , public interface::Implements<ConceptGetDev, QueueAlpaka3CpuNonBlocking<TDev>>
    {
    public:
        using QueueType = onHost::Queue<typename TDev::DevType, queueKind::NonBlocking>;

        auto getAlpaka3() const
        {
            return m_queueImpl;
        }

        explicit QueueAlpaka3CpuNonBlocking(TDev const& dev)
            : m_queueImpl(dev.getAlpaka3().makeQueue(queueKind::nonBlocking))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;
        }

        auto operator==(QueueAlpaka3CpuNonBlocking<TDev> const& rhs) const -> bool
        {
            return (m_queueImpl == rhs.m_queueImpl);
        }

        auto operator!=(QueueAlpaka3CpuNonBlocking<TDev> const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

    public:
        TDev m_dev;
        QueueType m_queueImpl;
    };

    namespace trait
    {
        //! The CPU non-blocking device queue device type trait specialization.
        template<typename TDev>
        struct DevType<QueueAlpaka3CpuNonBlocking<TDev>>
        {
            using type = TDev;
        };

        //! The CPU non-blocking device queue device get trait specialization.
        template<typename TDev>
        struct GetDev<QueueAlpaka3CpuNonBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto getDev(QueueAlpaka3CpuNonBlocking<TDev> const& queue) -> TDev
            {
                return queue.m_spQueueImpl.m_dev;
            }
        };

        //! The CPU non-blocking device queue event type trait specialization.
        template<typename TDev>
        struct EventType<QueueAlpaka3CpuNonBlocking<TDev>>
        {
            using type = EventAlpaka3Cpu<TDev>;
        };

        //! The CPU non-blocking device queue enqueue trait specialization.
        //! This default implementation for all tasks directly invokes the function call operator of the task.
        template<typename TDev, typename TTask>
        struct Enqueue<QueueAlpaka3CpuNonBlocking<TDev>, TTask>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueAlpaka3CpuNonBlocking<TDev>& queue, TTask const& task) -> void
            {
                queue.m_queueImpl.enqueueHostFn(task);
            }
        };

        //! The CPU non-blocking device queue test trait specialization.
        template<typename TDev>
        struct Empty<QueueAlpaka3CpuNonBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto empty(QueueAlpaka3CpuNonBlocking<TDev> const& queue) -> bool
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
        struct CurrentThreadWaitFor<QueueAlpaka3CpuNonBlocking<TDev>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueAlpaka3CpuNonBlocking<TDev> const& queue) -> void
            {
                auto q = queue.getAlpaka3();
                onHost::wait(q);
            }
        };
    } // namespace trait
} // namespace alpaka
