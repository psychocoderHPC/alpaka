/* Copyright 2025 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


// Specialized traits.
#include "aV2P/acc/AccAlpaka3Cpu.hpp"
#include "aV2P/acc/Traits.hpp"
#include "aV2P/atomic/Traits.hpp"
#include "aV2P/block/shared/dyn/Traits.hpp"
#include "aV2P/block/shared/st/Traits.hpp"
#include "aV2P/block/sync/Traits.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/platform/Traits.hpp"

// Implementation details.
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/Interface.hpp"

#ifdef __cpp_lib_format
#    include <format>
#endif
#include <memory>
#include <string>

namespace aV2P
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3;

    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    class DevAlpaka3;

    template<typename TDev, alpaka::concepts::QueueKind TQueueKind>
    class QueueAlpaka3;

    template<typename TDev>
    class EventAlpaka3Cpu;

    using DevCpu = DevAlpaka3<alpaka::onHost::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>>;


#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using DevCudaRt = DevAlpaka3<alpaka::onHost::DeviceSpec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>>;
#endif

    using QueueCpuBlocking = QueueAlpaka3<DevCpu, alpaka::queueKind::Blocking>;
    using QueueCpuNonBlocking = QueueAlpaka3<DevCpu, alpaka::queueKind::NonBlocking>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using QueueCudaRtBlocking = QueueAlpaka3<DevCudaRt, alpaka::queueKind::Blocking>;
    using QueueCudaRtNonBlocking = QueueAlpaka3<DevCudaRt, alpaka::queueKind::NonBlocking>;
#endif

    using EventCpu = EventAlpaka3Cpu<DevCpu>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using EventCudaRt = EventAlpaka3Cpu<DevCudaRt>;
#endif

    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuSerial, TDim, TIdx>
        {
            using type = aV2P::AccCpuSerial<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuSerial<TDim, TIdx>>
        {
            using type = aV2P::TagCpuSerial;
        };

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagCpuOmp2Blocks, TDim, TIdx>
        {
            using type = aV2P::AccCpuOmp2Blocks<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = aV2P::TagCpuOmp2Blocks;
        };
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagGpuCudaRt, TDim, TIdx>
        {
            using type = aV2P::AccGpuCudaRt<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccGpuCudaRt<TDim, TIdx>>
        {
            using type = aV2P::TagGpuCudaRt;
        };
#endif

        //! The CPU serial accelerator accelerator type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct AccType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct IsSingleThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
            : std::bool_constant<alpaka::exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct IsMultiThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
            : std::bool_constant<!alpaka::exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct GetAccDevProps<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
            ALPAKA_FN_HOST static auto getAccDevProps(DevAlpaka3<TDevSpec> const& dev) -> AccDevProps<TDim, TIdx>
            {
                auto prop = dev.getAlpaka3().getDeviceProperties();
                auto accProps = AccDevProps<TDim, TIdx>{// m_multiProcessorCount
                                                        static_cast<TIdx>(prop.multiProcessorCount),
                                                        // m_gridBlockExtentMax
                                                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                                                        // m_gridBlockCountMax
                                                        static_cast<TIdx>(std::numeric_limits<TIdx>::max()),
                                                        // m_blockThreadExtentMax
                                                        Vec<TDim, TIdx>::all(32u),
                                                        // m_blockThreadCountMax
                                                        static_cast<TIdx>(prop.maxThreadsPerBlock),
                                                        // m_threadElemExtentMax
                                                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                                                        // m_threadElemCountMax
                                                        std::numeric_limits<TIdx>::max(),
                                                        // m_sharedMemSizeBytes
                                                        prop.sharedMemPerBlockBytes,
                                                        // m_globalMemSizeBytes
                                                        prop.globalMemCapacityBytes};
                if constexpr(alpaka::exec::isSeqExecutor_v<TExecutor>)
                {
                    accProps.m_blockThreadCountMax = static_cast<TIdx>(1);
                    accProps.m_blockThreadExtentMax = Vec<TDim, TIdx>::all(1);
                }
                return accProps;
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct GetAccName<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return typename AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>::DevSpec{}.getName();
            }
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct DevType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using DevSpec = typename AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>::DevSpec;
            using type = DevAlpaka3<DevSpec>;
        };

        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct DimType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = TDim;
        };

        template<typename TKernel>
        struct DynSharedMemWrapper
        {
            TKernel kernel;
            uint32_t dynSharedMemBytes = 0u;

            constexpr DynSharedMemWrapper(TKernel const& kernelFunctor, uint32_t sharedMemBytes)
                : kernel{kernelFunctor}
                , dynSharedMemBytes{sharedMemBytes}
            {
            }

            constexpr void operator()(alpaka::onAcc::concepts::Acc auto const& acc, auto&&... args) const
            {
                kernel(acc, ALPAKA_FORWARD(args)...);
            }
        };

        //! The CPU serial accelerator execution task type trait specialization.
        template<
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                using Acc = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>;
                if(workDiv.m_blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error(
                        "The given work division is not valid for a single thread Acc: " + getAccName<Acc>()
                        + ". Threads per block should be 1!");
                }


                if constexpr(requires {
                                 // we need to check for the trait because the function signature is always available
                                 trait::BlockSharedMemDynSizeBytes<ALPAKA_TYPEOF(kernelFnObj), Acc>::
                                     getBlockSharedMemDynSizeBytes(
                                         kernelFnObj,
                                         workDiv.m_gridBlockExtent,
                                         workDiv.m_blockThreadExtent,
                                         std::forward<TArgs>(args)...);
                             })
                {
                    uint32_t dynSharedMemBytes = static_cast<uint32_t>(getBlockSharedMemDynSizeBytes<Acc>(
                        kernelFnObj,
                        workDiv.m_gridBlockExtent,
                        workDiv.m_blockThreadExtent,
                        std::forward<TArgs>(args)...));

                    // wrapp the kernel to provide the shared memroy as member for alpaka3
                    auto kernelWithSharedMem = DynSharedMemWrapper(kernelFnObj, dynSharedMemBytes);
                    return TaskKernelAlpaka3<Acc, TDim, TIdx, ALPAKA_TYPEOF(kernelWithSharedMem), TArgs...>(
                        workDiv,
                        kernelWithSharedMem,
                        std::forward<TArgs>(args)...);
                }
                else
                {
                    return TaskKernelAlpaka3<Acc, TDim, TIdx, TKernelFnObj, TArgs...>(
                        workDiv,
                        kernelFnObj,
                        std::forward<TArgs>(args)...);
                }
            }
        };

        //! The CPU serial execution task platform type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct PlatformType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using DevSpec = typename AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>::DevSpec;
            using type = PlatformAlpaka3<DevSpec>;
        };

        //! The CPU serial accelerator idx type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            alpaka::concepts::Api TApi,
            alpaka::concepts::DeviceKind TDeviceKind,
            alpaka::concepts::Executor TExecutor>
        struct IdxType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = TIdx;
        };

        template<typename TDict>
        struct DimType<alpaka::onAcc::Acc<TDict>>
        {
            using type = aV2P::DimInt<ALPAKA_TYPEOF(
                std::declval<alpaka::onAcc::Acc<TDict>>()[alpaka::layer::thread].idx())::dim()>;
        };

        template<typename TDict>
        struct IdxType<alpaka::onAcc::Acc<TDict>>
        {
            using type = ALPAKA_TYPEOF(std::declval<alpaka::onAcc::Acc<TDict>>()[alpaka::layer::thread].idx())::type;
        };

        template<typename TDict>
        struct GetWorkDiv<alpaka::onAcc::Acc<TDict>, origin::Grid, unit::Blocks>
        {
            //! \return The number of blocks in each dimension of the grid.
            static constexpr auto getWorkDiv(alpaka::onAcc::Acc<TDict> const& acc)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = Vec<Dim, Idx>;
                auto result = acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks);
                return VecType{result};
            }
        };

        template<typename TDict>
        struct GetWorkDiv<alpaka::onAcc::Acc<TDict>, origin::Block, unit::Threads>
        {
            //! \return The number of threads in each dimension of a block.
            static constexpr auto getWorkDiv(alpaka::onAcc::Acc<TDict> const& acc)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = Vec<Dim, Idx>;
                auto result = acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
                return VecType{result};
            }
        };

        template<typename TDict>
        struct GetWorkDiv<alpaka::onAcc::Acc<TDict>, origin::Thread, unit::Elems>
        {
            //! \return The number of elements in each dimension of the thread.
            static constexpr auto getWorkDiv(alpaka::onAcc::Acc<TDict> const& acc)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = Vec<Dim, Idx>;
                return VecType::all(1);
            }
        };

        template<typename TDict>
        struct GetIdx<alpaka::onAcc::Acc<TDict>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static constexpr auto getIdx(alpaka::onAcc::Acc<TDict> const& acc, TWorkDiv const&)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = Vec<Dim, Idx>;
                auto result = acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
                return VecType{result};
            }
        };

        template<typename TDict>
        struct GetIdx<alpaka::onAcc::Acc<TDict>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static constexpr auto getIdx(alpaka::onAcc::Acc<TDict> const& acc, TWorkDiv const&)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = Vec<Dim, Idx>;
                auto result = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks);
                return VecType{result};
            }
        };

        template<typename T, alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct GetDynSharedMem<T, TAlpakaAcc>
        {
            static auto getMem(TAlpakaAcc const& acc) -> T*
            {
                return alpaka::onAcc::getDynSharedMem<T>(acc);
            }
        };

        template<alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct SyncBlockThreads<TAlpakaAcc>
        {
            static constexpr auto syncBlockThreads(TAlpakaAcc const& acc) -> void
            {
                alpaka::onAcc::syncBlockThreads(acc);
            }
        };

        // shared mem
        template<typename T, std::size_t TuniqueId, alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct DeclareSharedVar<T, TuniqueId, TAlpakaAcc>
        {
            static constexpr auto declareVar(TAlpakaAcc const& acc) -> T&
            {
                return alpaka::onAcc::declareSharedVar<T, TuniqueId>(acc);
            }
        };

        template<alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct FreeSharedVars<TAlpakaAcc>
        {
            static constexpr auto freeVars(TAlpakaAcc const& acc) -> void
            {
            }
        };

        // atomic
        template<typename T, typename TAtomicOp, typename THierarchy, alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct AtomicOp<TAtomicOp, TAlpakaAcc, T, THierarchy>
        {
            static constexpr auto atomicOp(TAlpakaAcc const& acc, T* const addr, T const& value) -> T
            {
                // attention alpaka::onAcc::scope::system is currently not supported therefor we use device which will
                // be equal to alpaka v2
                if constexpr(std::same_as<THierarchy, hierarchy::Grids>)
                    return alpaka::onAcc::atomicOp<TAtomicOp>(acc, addr, value, alpaka::onAcc::scope::device);
                else if constexpr(std::same_as<THierarchy, hierarchy::Blocks>)
                    return alpaka::onAcc::atomicOp<TAtomicOp>(acc, addr, value, alpaka::onAcc::scope::device);
                else if constexpr(std::same_as<THierarchy, hierarchy::Threads>)
                    return alpaka::onAcc::atomicOp<TAtomicOp>(acc, addr, value, alpaka::onAcc::scope::block);

                ALPAKA_UNREACHABLE(T{});
            }

            static constexpr auto atomicOp(TAlpakaAcc const& acc, T* const addr, T const& compare, T const& value) -> T
            {
                // attention alpaka::onAcc::scope::system is currently not supported therefor we use device which will
                // be equal to alpaka v2
                if constexpr(std::same_as<THierarchy, hierarchy::Grids>)
                    return alpaka::onAcc::atomicOp<TAtomicOp>(acc, addr, compare, value, alpaka::onAcc::scope::device);
                else if constexpr(std::same_as<THierarchy, hierarchy::Blocks>)
                    return alpaka::onAcc::atomicOp<TAtomicOp>(acc, addr, compare, value, alpaka::onAcc::scope::device);
                else if constexpr(std::same_as<THierarchy, hierarchy::Threads>)
                    return alpaka::onAcc::atomicOp<TAtomicOp>(acc, addr, compare, value, alpaka::onAcc::scope::block);

                ALPAKA_UNREACHABLE(T{});
            }
        };
    } // namespace trait

    namespace interface
    {
        //! Checks whether the interface is implemented by the given class
        template<typename TDict, typename TInterface>
        struct ImplementsInterface<TInterface, alpaka::onAcc::Acc<TDict>>
        {
            static constexpr auto value = true;
        };

        namespace detail
        {
            template<typename TDict, typename TInterface>
            struct ImplementationBaseType<
                TInterface,
                alpaka::onAcc::Acc<TDict>,
                std::enable_if_t<ImplementsInterface<TInterface, alpaka::onAcc::Acc<TDict>>::value>>
            {
                using type = alpaka::onAcc::Acc<TDict>;
            };
        } // namespace detail
    } // namespace interface
} // namespace aV2P
