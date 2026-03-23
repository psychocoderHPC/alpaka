/* Copyright 2025 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


// Specialized traits.

#include "aV2P/acc/Traits.hpp"

namespace aV2P
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3;

    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3;

    //! The CPU serial accelerator.
    //!
    //! This accelerator allows serial kernel execution on a CPU device.
    //! The block idx is restricted to 1x1x1 and all blocks are executed serially so there is no parallelism at all.
    template<
        typename TDim,
        typename TIdx,
        alpaka::concepts::Api TApi,
        alpaka::concepts::DeviceKind TDeviceKind,
        alpaka::concepts::Executor TExecutor>
    class AccAlpaka3 final
        : public interface::Implements<InterfaceAcc, AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TAcc, typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::aV2P::TaskKernelAlpaka3;

        using DevSpec = alpaka::onHost::DeviceSpec<TApi, TDeviceKind>;
        using Executor = TExecutor;
        using Dim = TDim;
        using Idx = TIdx;

        AccAlpaka3(AccAlpaka3 const&) = delete;
        AccAlpaka3(AccAlpaka3&&) = delete;
        auto operator=(AccAlpaka3 const&) -> AccAlpaka3& = delete;
        auto operator=(AccAlpaka3&&) -> AccAlpaka3& = delete;
    };

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuSerial = AccAlpaka3<TDim, TIdx, alpaka::api::Host, alpaka::deviceKind::Cpu, alpaka::exec::CpuSerial>;
#endif
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    using AccCpuOmp2Blocks
        = AccAlpaka3<TDim, TIdx, alpaka::api::Host, alpaka::deviceKind::Cpu, alpaka::exec::CpuOmpBlocks>;
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    using AccGpuCudaRt
        = AccAlpaka3<TDim, TIdx, alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu, alpaka::exec::GpuCuda>;
#endif

} // namespace aV2P

#include "aV2P/acc/AccAlpaka3Traits.hpp"
