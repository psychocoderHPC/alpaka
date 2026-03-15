/* Copyright 2025 Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci,
 * Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/aV2P.hpp"

namespace aV2P::test
{
    namespace trait
    {
        //! The default queue type trait for devices.
        template<typename TDev, typename TSfinae = void>
        struct DefaultQueueType;

        //! The default queue type trait specialization for the CPU device.
        template<>
        struct DefaultQueueType<DevCpu>
        {
#if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = QueueCpuBlocking;
#else
            using type = QueueCpuNonBlocking;
#endif
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

        //! The default queue type trait specialization for the CUDA/HIP device.
        template<typename TApi>
        struct DefaultQueueType<DevUniformCudaHipRt<TApi>>
        {
#    if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = QueueUniformCudaHipRtBlocking<TApi>;
#    else
            using type = QueueUniformCudaHipRtNonBlocking<TApi>;
#    endif
        };
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
        //! The default queue type trait specialization for the SYCL device.
        template<concepts::Tag TTag>
        struct DefaultQueueType<DevGenericSycl<TTag>>
        {
#    if(ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            using type = QueueGenericSyclBlocking<TTag>;
#    else
            using type = QueueGenericSyclNonBlocking<TTag>;
#    endif
        };
#endif

        //! The blocking queue trait.
        template<typename TQueue, typename TSfinae = void>
        struct IsBlockingQueue;

        //! The blocking queue trait specialization for a blocking CPU queue.
        template<typename TDev>
        struct IsBlockingQueue<QueueGenericThreadsBlocking<TDev>>
        {
            static constexpr bool value = true;
        };

        //! The blocking queue trait specialization for a non-blocking CPU queue.
        template<typename TDev>
        struct IsBlockingQueue<QueueGenericThreadsNonBlocking<TDev>>
        {
            static constexpr bool value = false;
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

        //! The blocking queue trait specialization for a blocking CUDA/HIP RT queue.
        template<typename TApi>
        struct IsBlockingQueue<QueueUniformCudaHipRtBlocking<TApi>>
        {
            static constexpr bool value = true;
        };

        //! The blocking queue trait specialization for a non-blocking CUDA/HIP RT queue.
        template<typename TApi>
        struct IsBlockingQueue<QueueUniformCudaHipRtNonBlocking<TApi>>
        {
            static constexpr bool value = false;
        };
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
        template<concepts::Tag TTag>
        struct IsBlockingQueue<QueueGenericSyclBlocking<TTag>>
        {
            static constexpr auto value = true;
        };

        template<concepts::Tag TTag>
        struct IsBlockingQueue<QueueGenericSyclNonBlocking<TTag>>
        {
            static constexpr auto value = false;
        };
#endif
    } // namespace trait

    //! The queue type that should be used for the given device.
    template<typename TDev>
    using DefaultQueue = typename trait::DefaultQueueType<TDev>::type;

    //! The queue type that should be used for the given accelerator.
    template<typename TQueue>
    using IsBlockingQueue = trait::IsBlockingQueue<TQueue>;

    //! A std::tuple holding tuples of devices and corresponding queue types.
    using TestQueues = std::tuple<
        std::tuple<DevCpu, QueueCpuBlocking>,
        std::tuple<DevCpu, QueueCpuNonBlocking>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        ,
        std::tuple<DevCudaRt, QueueCudaRtBlocking>,
        std::tuple<DevCudaRt, QueueCudaRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        ,
        std::tuple<DevHipRt, QueueHipRtBlocking>,
        std::tuple<DevHipRt, QueueHipRtNonBlocking>
#endif
#ifdef ALPAKA_ACC_SYCL_ENABLED
#    ifdef ALPAKA_SYCL_ONEAPI_CPU
        ,
        std::tuple<aV2P::DevCpuSycl, aV2P::QueueCpuSyclBlocking>,
        std::tuple<aV2P::DevCpuSycl, aV2P::QueueCpuSyclNonBlocking>
#    endif
#    ifdef ALPAKA_SYCL_ONEAPI_FPGA
        ,
        std::tuple<aV2P::DevFpgaSyclIntel, aV2P::QueueFpgaSyclIntelBlocking>,
        std::tuple<aV2P::DevFpgaSyclIntel, aV2P::QueueFpgaSyclIntelNonBlocking>
#    endif
#    ifdef ALPAKA_SYCL_ONEAPI_GPU
        ,
        std::tuple<aV2P::DevGpuSyclIntel, aV2P::QueueGpuSyclIntelBlocking>,
        std::tuple<aV2P::DevGpuSyclIntel, aV2P::QueueGpuSyclIntelNonBlocking>
#    endif
#    ifdef ALPAKA_SYCL_ONEAPI_GPU_NVIDIA
        ,
        std::tuple<aV2P::DevGpuSyclNvidia, aV2P::QueueGpuSyclNvidiaBlocking>,
        std::tuple<aV2P::DevGpuSyclNvidia, aV2P::QueueGpuSyclNvidiaNonBlocking>
#    endif
#    ifdef ALPAKA_SYCL_ONEAPI_GPU_AMD
        ,
        std::tuple<aV2P::DevGpuSyclAmd, aV2P::QueueGpuSyclAmdBlocking>,
        std::tuple<aV2P::DevGpuSyclAmd, aV2P::QueueGpuSyclAmdNonBlocking>
#    endif
#endif
        >;
} // namespace aV2P::test
