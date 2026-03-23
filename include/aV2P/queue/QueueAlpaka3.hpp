/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccAlpaka3Cpu.hpp"
#include "aV2P/core/Config.hpp"
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
    template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
    class QueueAlpaka3 final
        : public interface::Implements<ConceptCurrentThreadWaitFor, QueueAlpaka3<TDev, TQueueKind>>
        , public interface::Implements<ConceptQueue, QueueAlpaka3<TDev, TQueueKind>>
        , public interface::Implements<ConceptGetDev, QueueAlpaka3<TDev, TQueueKind>>
    {
    public:
        using QueueType = alpaka::onHost::Queue<typename TDev::DevType, alpaka::queueKind::Blocking>;

        auto getAlpaka3() const
        {
            return m_queueImpl;
        }

        explicit QueueAlpaka3(TDev const& dev) : m_queueImpl{dev.getAlpaka3().makeQueue(alpaka::queueKind::blocking)}
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;
        }

        auto operator==(QueueAlpaka3<TDev, TQueueKind> const& rhs) const -> bool
        {
            return (m_queueImpl == rhs.m_queueImpl);
        }

        auto operator!=(QueueAlpaka3<TDev, TQueueKind> const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

    public:
        QueueType m_queueImpl;
    };

    namespace trait
    {
        //! The CPU non-blocking device queue device type trait specialization.
        template<typename TDev, typename TQueueKind>
        struct DevType<QueueAlpaka3<TDev, TQueueKind>>
        {
            using type = TDev;
        };

        //! The CPU non-blocking device queue device get trait specialization.
        template<typename TDev, typename TQueueKind>
        struct GetDev<QueueAlpaka3<TDev, TQueueKind>>
        {
            ALPAKA_FN_HOST static auto getDev(QueueAlpaka3<TDev, TQueueKind> const& queue) -> TDev
            {
                return TDev{queue.getAlpaka3().getDevice()};
            }
        };

        //! The CPU non-blocking device queue event type trait specialization.
        template<typename TDev, typename TQueueKind>
        struct EventType<QueueAlpaka3<TDev, TQueueKind>>
        {
            using type = EventAlpaka3Cpu<TDev>;
        };

        //! The CPU non-blocking device queue enqueue trait specialization.
        //! This default implementation for all tasks directly invokes the function call operator of the task.
        template<typename TDev, typename TTask, typename TQueueKind>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, TTask>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueAlpaka3<TDev, TQueueKind>& queue, TTask const& task) -> void
            {
                queue.getAlpaka3().enqueueHostFn(task);
            }
        };

        //! The CPU non-blocking device queue test trait specialization.
        template<typename TDev, typename TQueueKind>
        struct Empty<QueueAlpaka3<TDev, TQueueKind>>
        {
            ALPAKA_FN_HOST static auto empty(QueueAlpaka3<TDev, TQueueKind> const& queue) -> bool
            {
                static_assert(sizeof(TDev) && false, "Empty<Queue> Not implemented");
                return true;
            }
        };

        //! The CPU blocking device queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<typename TDev, typename TQueueKind>
        struct CurrentThreadWaitFor<QueueAlpaka3<TDev, TQueueKind>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueAlpaka3<TDev, TQueueKind> const& queue) -> void
            {
                auto q = queue.getAlpaka3();
                alpaka::onHost::wait(q);
            }
        };
    } // namespace trait
} // namespace aV2P
