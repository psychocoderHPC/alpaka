/* Copyright 2022 Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace alpaka::core
{
    class CallbackThread
    {
        using Task = std::packaged_task<void()>;

    public:
        ~CallbackThread()
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            {
                std::unique_lock<std::mutex> lock{m_mutex};
                m_stop = true;
                m_cond.notify_one();
            }

            if(m_thread.joinable())
            {
                if(std::this_thread::get_id() == m_thread.get_id())
                {
                    std::cerr << "ERROR in ~CallbackThread: thread joins itself" << std::endl;
                    std::abort();
                }
                m_thread.join();
            }
        }
        auto submit(Task&& newTask) -> std::future<void>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            assert(!m_stop);
            auto f = newTask.get_future();
            {
                std::unique_lock<std::mutex> lock{m_mutex};

                m_tasks.emplace(std::move(newTask));
                ++m_tasksInProgress;
                if(!m_thread.joinable())
                    startWorkerThread();
                m_cond.notify_one();
            }

            return f;
        }

        template<typename F>
        auto submit(F&& f) -> std::future<void>
        {
            return submit(Task{std::forward<F>(f)});
        }

        bool empty()
        {
            // thread sanitizer will show errors if this lock is not here
            std::unique_lock<std::mutex> lock{m_mutex};
            return m_tasksInProgress.load() == 0;
        }

    private:
        std::thread m_thread;
        std::condition_variable m_cond;
        std::mutex m_mutex;
        bool m_stop{false};
        std::queue<Task> m_tasks{};
        std::atomic<int> m_tasksInProgress{0};

        auto startWorkerThread() -> void
        {
            m_thread = std::thread(
                [this]
                {
                    while(true)
                    {
                        Task task;
                        {
                            std::unique_lock<std::mutex> lock{m_mutex};
                            m_cond.wait(lock, [this] { return m_stop || m_tasksInProgress; });

                            if(m_stop && m_tasks.empty())
                                break;

                            task = std::move(m_tasks.front());
                            m_tasks.pop();
                        }
                        task();
                        --m_tasksInProgress;
                    }
                });
        }
    };
} // namespace alpaka::core
