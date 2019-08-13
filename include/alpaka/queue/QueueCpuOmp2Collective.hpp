/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/QueueCpuBlocking.hpp>
#include <alpaka/wait/Traits.hpp>

#include <atomic>
#include <mutex>

namespace alpaka
{
    namespace event
    {
        class EventCpu;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU collective device queue implementation.
                class QueueCpuOmp2CollectiveImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    QueueCpuOmp2CollectiveImpl(
                        dev::DevCpu const & dev) noexcept :
                            m_dev(dev),
                            m_uCurrentlyExecutingTask(0u)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueCpuOmp2CollectiveImpl(QueueCpuOmp2CollectiveImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCpuOmp2CollectiveImpl(QueueCpuOmp2CollectiveImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuOmp2CollectiveImpl const &) -> QueueCpuOmp2CollectiveImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuOmp2CollectiveImpl &&) -> QueueCpuOmp2CollectiveImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ~QueueCpuOmp2CollectiveImpl() = default;

                public:
                    dev::DevCpu const m_dev;            //!< The device this queue is bound to.
                    std::mutex mutable m_mutex;
                    std::atomic<uint32_t> m_uCurrentlyExecutingTask;
                };
            }
        }

        //#############################################################################
        //! The CPU collective device queue.
        //
        // @attention queue can only be used together with the accelerator AccCpuOmp2Blocks
        //
        class QueueCpuOmp2Collective final
        {
        public:
            //-----------------------------------------------------------------------------
            QueueCpuOmp2Collective(
                dev::DevCpu const & dev) :
                    m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuOmp2CollectiveImpl>(dev)),
                    m_spBlockingQueue(std::make_shared<QueueCpuBlocking>(dev))
            {
                dev.m_spDevCpuImpl->RegisterOmp2CollectiveQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueCpuOmp2Collective(QueueCpuOmp2Collective const &) = default;
            //-----------------------------------------------------------------------------
            QueueCpuOmp2Collective(QueueCpuOmp2Collective &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuOmp2Collective const &) -> QueueCpuOmp2Collective & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuOmp2Collective &&) -> QueueCpuOmp2Collective & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueCpuOmp2Collective const & rhs) const
            -> bool
            {
                return m_spQueueImpl == rhs.m_spQueueImpl && m_spBlockingQueue == rhs.m_spBlockingQueue;
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueCpuOmp2Collective const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCpuOmp2Collective() = default;

        public:
            std::shared_ptr<cpu::detail::QueueCpuOmp2CollectiveImpl> m_spQueueImpl;
            std::shared_ptr<QueueCpuBlocking> m_spBlockingQueue;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCpuOmp2Collective>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU blocking device queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCpuOmp2Collective>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCpuOmp2Collective const & queue)
                -> dev::DevCpu
                {
                    return queue.m_spQueueImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueCpuOmp2Collective>
            {
                using type = event::EventCpu;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {

            //#############################################################################
            //! The CPU blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueCpuOmp2Collective,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuOmp2Collective & queue,
                    TTask const & task)
                -> void
                {
                    if(::omp_in_parallel() != 0)
                    {
                        // wait for all tasks en-queued before the parallel region
                        while(!queue::empty(*queue.m_spBlockingQueue)){}
                        queue.m_spQueueImpl->m_uCurrentlyExecutingTask += 1u;

                        #pragma omp single nowait
                        task();

                        queue.m_spQueueImpl->m_uCurrentlyExecutingTask -= 1u;
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                        queue::enqueue(*queue.m_spBlockingQueue, task);
                    }
                }
            };

            //#############################################################################
            //! The CPU blocking device queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueCpuOmp2Collective>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCpuOmp2Collective const & queue)
                -> bool
                {
                    return queue.m_spQueueImpl->m_uCurrentlyExecutingTask == 0u &&
                        queue::empty(*queue.m_spBlockingQueue);
                }
            };
        }
    }

    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCpuOmp2Collective>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCpuOmp2Collective const & queue)
                -> void
                {
                    if(::omp_in_parallel() != 0)
                    {
                        // wait for all tasks en-queued before the parallel region
                        while(!queue::empty(*queue.m_spBlockingQueue)){}
                        #pragma omp barrier
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                        wait::wait(*queue.m_spBlockingQueue);
                    }
                }
            };
        }
    }
}

#endif
