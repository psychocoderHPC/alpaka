/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Specialized traits.
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/platform/Traits.hpp"

// Implementation details.
#include "alpaka/acc/AccAlpaka3Cpu.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/dev/DevAlpaka3Cpu.hpp"
#include "alpaka/kernel/KernelFunctionAttributes.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace alpaka
{
    class DevAlpaka3Cpu;

    //! The CPU serial execution task implementation.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3 final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelAlpaka3(TWorkDiv&& workDiv, TKernelFnObj kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelBundle(kernelFnObj, std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        KernelBundle<TKernelFnObj, TArgs...> m_kernelBundle;
    };

    namespace trait
    {
        //! The CPU serial execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccAlpaka3Cpu<TDim, TIdx>;
        };

        //! The CPU serial execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevAlpaka3Cpu;
        };

        //! The CPU serial execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The CPU serial execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PlatformType<TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PlatformAlpaka3Cpu;
        };

        //! The CPU serial execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };

        //! \brief Specialisation of the class template FunctionAttributes
        //! \tparam TDev The device type.
        //! \tparam TDim The dimensionality of the accelerator device properties.
        //! \tparam TIdx The idx type of the accelerator device properties.
        //! \tparam TKernelFn Kernel function object type.
        //! \tparam TArgs Kernel function object argument types as a parameter pack.
        template<typename TDev, typename TDim, typename TIdx, typename TKernelFn, typename... TArgs>
        struct FunctionAttributes<AccAlpaka3Cpu<TDim, TIdx>, TDev, TKernelFn, TArgs...>
        {
            //! \param dev The device instance
            //! \param kernelFn The kernel function object which should be executed.
            //! \param args The kernel invocation arguments.
            //! \return KernelFunctionAttributes instance. The default version always returns an instance with zero
            //! fields. For CPU, the field of max threads allowed by kernel function for the block is 1.
            ALPAKA_FN_HOST static auto getFunctionAttributes(
                TDev const& dev,
                [[maybe_unused]] TKernelFn const& kernelFn,
                [[maybe_unused]] TArgs&&... args) -> alpaka::KernelFunctionAttributes
            {
                alpaka::KernelFunctionAttributes kernelFunctionAttributes;
                kernelFunctionAttributes.maxThreadsPerBlock
                    = dev.getAlpaka3().getDeviceProperties().maxThreadsPerBlock;
                kernelFunctionAttributes.maxDynamicSharedSizeBytes
                    = dev.getAlpaka3().getDeviceProperties().sharedMemPerBlockBytes;

                return kernelFunctionAttributes;
            }
        };

        template<typename TDev, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<QueueAlpaka3CpuNonBlocking<TDev>, TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3CpuNonBlocking<TDev>& queue,
                TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                auto tSpec
                    = onHost::ThreadSpec{task.m_gridBlockExtent.getAlpaka3(), task.m_blockThreadExtent.getAlpaka3()};
                queue.getAlpaka3().enqueue(exec::cpuSerial, tSpec, task.m_kernelBundle);
            }
        };

        template<typename TDev, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<QueueAlpaka3CpuBlocking<TDev>, TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3CpuBlocking<TDev>& queue,
                TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                auto tSpec
                    = onHost::ThreadSpec{task.m_gridBlockExtent.getAlpaka3(), task.m_blockThreadExtent.getAlpaka3()};
                queue.getAlpaka3().enqueue(exec::cpuSerial, tSpec, task.m_kernelBundle);
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
