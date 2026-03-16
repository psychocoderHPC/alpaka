/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Assert.hpp"
#include "aV2P/core/Utility.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/event/Traits.hpp"
#include "aV2P/queue/QueueAlpaka3.hpp"
#include "aV2P/wait/Traits.hpp"

#include <condition_variable>
#include <future>
#include <mutex>
#include <utility>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

namespace aV2P
{
    //! The CPU device event.
    template<typename TDev>
    class EventAlpaka3Cpu final
        : public interface::Implements<ConceptCurrentThreadWaitFor, EventAlpaka3Cpu<TDev>>
        , public interface::Implements<ConceptGetDev, EventAlpaka3Cpu<TDev>>
    {
    public:
        using EventType = decltype(std::declval<TDev>().m_devImpl.makeEvent());

        //! \param bBusyWaiting Unused. EventAlpaka3Cpu never does busy waiting.
        EventAlpaka3Cpu(TDev const& dev, [[maybe_unused]] bool bBusyWaiting = true)
            : m_dev(dev)
            , m_eventImpl(dev.makeEvent())
        {
        }

        auto operator==(EventAlpaka3Cpu<TDev> const& rhs) const -> bool
        {
            return (m_eventImpl == rhs.m_eventImpl);
        }

        auto operator!=(EventAlpaka3Cpu<TDev> const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

    public:
        TDev const m_dev;
        EventType m_eventImpl;
    };

    using EventCpu = EventAlpaka3Cpu<DevCpu>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using EventCudaRt = EventAlpaka3Cpu<DevCudaRt>;
#endif

    namespace trait
    {
        //! The CPU device event device type trait specialization.
        template<typename TDev>
        struct DevType<EventAlpaka3Cpu<TDev>>
        {
            using type = TDev;
        };

        //! The CPU device event device get trait specialization.
        template<typename TDev>
        struct GetDev<EventAlpaka3Cpu<TDev>>
        {
            ALPAKA_FN_HOST static auto getDev(EventAlpaka3Cpu<TDev> const& event) -> TDev
            {
                return event.m_dev;
            }
        };

        //! The CPU device event test trait specialization.
        template<typename TDev>
        struct IsComplete<EventAlpaka3Cpu<TDev>>
        {
            //! \return If the event is not waiting within a queue (not enqueued or already handled).
            ALPAKA_FN_HOST static auto isComplete(EventAlpaka3Cpu<TDev> const& event) -> bool
            {
                return event.m_eventImpl.isComplete();
            }
        };

        //! The CPU non-blocking device queue enqueue trait specialization.
        template<typename TDev, typename TQueueKind>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, EventAlpaka3Cpu<TDev>>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueAlpaka3<TDev, TQueueKind>& queue, EventAlpaka3Cpu<TDev>& event)
                -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                queue.m_queueImpl.enqueue(event.m_queueImpl);
            }
        };
    } // namespace trait

    namespace trait
    {
        //! The CPU device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been
        //! completed. If the event is not enqueued to a queue the method returns immediately.
        template<typename TDev>
        struct CurrentThreadWaitFor<EventAlpaka3Cpu<TDev>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventAlpaka3Cpu<TDev> const& event) -> void
            {
                alpaka::onHost::wait(event.m_eventImpl);
            }
        };

        //! The CPU non-blocking device queue event wait trait specialization.
        template<typename TDev, typename TQueueKind>
        struct WaiterWaitFor<QueueAlpaka3<TDev, TQueueKind>, EventAlpaka3Cpu<TDev>>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                EventAlpaka3Cpu<TDev> const& event) -> void
            {
                queue.getAlpaka3().waitFor(event.m_eventImpl);
            }
        };
    } // namespace trait
} // namespace aV2P
