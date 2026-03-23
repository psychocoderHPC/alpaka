/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Specialized traits.
#include "aV2P/acc/Traits.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/platform/Traits.hpp"

// Implementation details.
#include "aV2P/acc/AccAlpaka3Cpu.hpp"
#include "aV2P/core/Decay.hpp"
#include "aV2P/dev/DevAlpaka3Cpu.hpp"
#include "aV2P/kernel/KernelFunctionAttributes.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/meta/NdLoop.hpp"
#include "aV2P/platform/PlatformAlpaka3Cpu.hpp"
#include "aV2P/workdiv/WorkDivMembers.hpp"

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace aV2P
{
    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    class DevAlpaka3;

    //! The CPU serial execution task implementation.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3 final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        auto getExecutor() const
        {
            return typename TAcc::Executor{};
        }

        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelAlpaka3(TWorkDiv&& workDiv, TKernelFnObj kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelBundle(kernelFnObj, std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        alpaka::KernelBundle<TKernelFnObj, TArgs...> m_kernelBundle;
    };

    namespace trait
    {
        //! The CPU serial execution task accelerator type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelAlpaka3<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TAcc;
        };

        //! The CPU serial execution task device type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelAlpaka3<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using DevSpec = typename TAcc::DevSpec;
            using type = DevAlpaka3<DevSpec>;
        };

        //! The CPU serial execution task dimension getter trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelAlpaka3<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The CPU serial execution task platform type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PlatformType<TaskKernelAlpaka3<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using DevSpec = typename TAcc::DevSpec;
            using type = PlatformAlpaka3<DevSpec>;
        };

        //! The CPU serial execution task idx type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelAlpaka3<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };

        //! \brief Specialisation of the class template FunctionAttributes
        //! \tparam TDev The device type.
        //! \tparam TDim The dimensionality of the accelerator device properties.
        //! \tparam TIdx The idx type of the accelerator device properties.
        //! \tparam TKernelFn Kernel function object type.
        //! \tparam TArgs Kernel function object argument types as a parameter pack.
        template<
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TDev,
            typename TDim,
            typename TIdx,
            typename TKernelFn,
            typename... TArgs>
        struct FunctionAttributes<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>, TDev, TKernelFn, TArgs...>
        {
            //! \param dev The device instance
            //! \param kernelFn The kernel function object which should be executed.
            //! \param args The kernel invocation arguments.
            //! \return KernelFunctionAttributes instance. The default version always returns an instance with zero
            //! fields. For CPU, the field of max threads allowed by kernel function for the block is 1.
            ALPAKA_FN_HOST static auto getFunctionAttributes(
                TDev const& dev,
                [[maybe_unused]] TKernelFn const& kernelFn,
                [[maybe_unused]] TArgs&&... args) -> aV2P::KernelFunctionAttributes
            {
                aV2P::KernelFunctionAttributes kernelFunctionAttributes;
                kernelFunctionAttributes.maxThreadsPerBlock
                    = static_cast<TIdx>(dev.getAlpaka3().getDeviceProperties().maxThreadsPerBlock);
                kernelFunctionAttributes.maxDynamicSharedSizeBytes
                    = static_cast<TIdx>(dev.getAlpaka3().getDeviceProperties().sharedMemPerBlockBytes);

                if constexpr(alpaka::exec::isSeqExecutor_v<TExecutor>)
                {
                    kernelFunctionAttributes.maxThreadsPerBlock = static_cast<TIdx>(1);
                }

                return kernelFunctionAttributes;
            }
        };

        template<
            typename TDev,
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename TQueueKind,
            typename... TArgs>
        struct Enqueue<QueueAlpaka3<TDev, TQueueKind>, TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueAlpaka3<TDev, TQueueKind>& queue,
                TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                auto tSpec = alpaka::onHost::ThreadSpec{
                    task.m_gridBlockExtent.getAlpaka3(),
                    task.m_blockThreadExtent.getAlpaka3()};
                queue.getAlpaka3().enqueue(task.getExecutor(), tSpec, task.m_kernelBundle);
            }
        };
    } // namespace trait
} // namespace aV2P

#endif
