/* Copyright 2025 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "alpaka/dev/DevAlpaka3Cpu.hpp"

// Specialized traits.
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/Traits.hpp"

// Implementation details.
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/dev/DevAlpaka3Cpu.hpp"

#ifdef __cpp_lib_format
#    include <format>
#endif
#include <memory>
#include <string>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace alpaka
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3;

    //! The CPU serial accelerator.
    //!
    //! This accelerator allows serial kernel execution on a CPU device.
    //! The block idx is restricted to 1x1x1 and all blocks are executed serially so there is no parallelism at all.
    template<
        typename TDim,
        typename TIdx,
        concepts::Api TApi,
        concepts::DeviceKind TDeviceKind,
        concepts::Executor TExecutor>
    class AccAlpaka3 final
        : public interface::Implements<InterfaceAcc, AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelAlpaka3;

        using DevSpec = onHost::DeviceSpec<TApi, TDeviceKind>;
        using Executor = TExecutor;
        using Dim = TDim;
        using Idx = TIdx;

        AccAlpaka3(AccAlpaka3 const&) = delete;
        AccAlpaka3(AccAlpaka3&&) = delete;
        auto operator=(AccAlpaka3 const&) -> AccAlpaka3& = delete;
        auto operator=(AccAlpaka3&&) -> AccAlpaka3& = delete;
    };

    template<typename TDim, typename TIdx>
    using AccAlpaka3Cpu = AccAlpaka3<TDim, TIdx, api::Host, deviceKind::Cpu, exec::CpuSerial>;

    template<typename TDim, typename TIdx>
    using AccAlpaka3Omp = AccAlpaka3<TDim, TIdx, api::Host, deviceKind::Cpu, exec::CpuOmpBlocks>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagAlpaka3Cpu, TDim, TIdx>
        {
            using type = alpaka::AccAlpaka3Cpu<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = alpaka::TagAlpaka3Cpu;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagAlpaka3Omp, TDim, TIdx>
        {
            using type = alpaka::AccAlpaka3Omp<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccAlpaka3Omp<TDim, TIdx>>
        {
            using type = alpaka::TagAlpaka3Omp;
        };

        //! The CPU serial accelerator accelerator type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct AccType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>;
        };

        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct IsSingleThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
            : std::bool_constant<exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct IsMultiThreadAcc<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
            : std::bool_constant<!exec::isSeqExecutor_v<TExecutor>>
        {
        };

        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct GetAccDevProps<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            template<onHost::concepts::DeviceSpec TDevSpec>
            ALPAKA_FN_HOST static auto getAccDevProps(DevAlpaka3<TDevSpec> const& dev) -> AccDevProps<TDim, TIdx>
            {
                auto prop = dev.getAlpaka3().getDeviceProperties();
                return {// m_multiProcessorCount
                        prop.multiProcessorCount,
                        // m_gridBlockExtentMax
                        VecOld<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        prop.maxThreadsPerBlock,
                        // m_blockThreadExtentMax
                        VecOld<TDim, TIdx>::ones(),
                        // m_blockThreadCountMax
                        static_cast<TIdx>(1),
                        // m_threadElemExtentMax
                        VecOld<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        prop.sharedMemPerBlockBytes,
                        // m_globalMemSizeBytes
                        prop.globalMemCapacityBytes};
            }
        };

        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
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
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct DevType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using DevSpec = typename AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>::DevSpec;
            using type = DevAlpaka3<DevSpec>;
        };

        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct DimType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = TDim;
        };

        //! The CPU serial accelerator execution task type trait specialization.
        template<
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor,
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
                if(workDiv.m_blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error(
                        "The given work division is not valid for a single thread Acc: "
                        + getAccName<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>()
                        + ". Threads per block should be 1!");
                }

                return TaskKernelAlpaka3<
                    AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>,
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs...>(workDiv, kernelFnObj, std::forward<TArgs>(args)...);
            }
        };

        //! The CPU serial execution task platform type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct PlatformType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using DevSpec = typename AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>::DevSpec;
            using type = PlatformAlpaka3<DevSpec>;
        };

        //! The CPU serial accelerator idx type trait specialization.
        template<
            typename TDim,
            typename TIdx,
            concepts::Api TApi,
            concepts::DeviceKind TDeviceKind,
            concepts::Executor TExecutor>
        struct IdxType<AccAlpaka3<TDim, TIdx, TApi, TDeviceKind, TExecutor>>
        {
            using type = TIdx;
        };

        template<typename TDict>
        struct DimType<onAcc::Acc<TDict>>
        {
            using type = alpaka::DimInt<ALPAKA_TYPEOF(std::declval<onAcc::Acc<TDict>>()[layer::thread].idx())::dim()>;
        };

        template<typename TDict>
        struct IdxType<onAcc::Acc<TDict>>
        {
            using type = ALPAKA_TYPEOF(std::declval<onAcc::Acc<TDict>>()[layer::thread].idx())::type;
        };

        template<typename TDict>
        struct GetWorkDiv<onAcc::Acc<TDict>, origin::Grid, unit::Blocks>
        {
            //! \return The number of blocks in each dimension of the grid.
            static constexpr auto getWorkDiv(onAcc::Acc<TDict> const& acc)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = VecOld<Dim, Idx>;
                auto result = acc.getExtentsOf(onAcc::origin::grid, onAcc::unit::blocks);
                return VecType{result};
            }
        };

        template<typename TDict>
        struct GetWorkDiv<onAcc::Acc<TDict>, origin::Block, unit::Threads>
        {
            //! \return The number of threads in each dimension of a block.
            static constexpr auto getWorkDiv(onAcc::Acc<TDict> const& acc)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = VecOld<Dim, Idx>;
                auto result = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
                return VecType{result};
            }
        };

        template<typename TDict>
        struct GetWorkDiv<onAcc::Acc<TDict>, origin::Thread, unit::Elems>
        {
            //! \return The number of elements in each dimension of the thread.
            static constexpr auto getWorkDiv(onAcc::Acc<TDict> const& acc)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = VecOld<Dim, Idx>;
                return VecType::all(1);
            }
        };

        template<typename TDict>
        struct GetIdx<onAcc::Acc<TDict>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static constexpr auto getIdx(onAcc::Acc<TDict> const& acc, TWorkDiv const& workDiv)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = VecOld<Dim, Idx>;
                auto result = acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads);
                return VecType{result};
            }
        };

        template<typename TDict>
        struct GetIdx<onAcc::Acc<TDict>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static constexpr auto getIdx(onAcc::Acc<TDict> const& acc, TWorkDiv const&)
            {
                using Dim = typename DimType<ALPAKA_TYPEOF(acc)>::type;
                using Idx = typename IdxType<ALPAKA_TYPEOF(acc)>::type;
                using VecType = VecOld<Dim, Idx>;
                auto result = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::blocks);
                return VecType{result};
            }
        };
    } // namespace trait

    namespace interface
    {
        //! Checks whether the interface is implemented by the given class
        template<typename TDict, typename TInterface>
        struct ImplementsInterface<TInterface, onAcc::Acc<TDict>>
        {
            static constexpr auto value = true;
        };

        namespace detail
        {
            template<typename TDict, typename TInterface>
            struct ImplementationBaseType<
                TInterface,
                onAcc::Acc<TDict>,
                std::enable_if_t<ImplementsInterface<TInterface, onAcc::Acc<TDict>>::value>>
            {
                using type = onAcc::Acc<TDict>;
            };
        } // namespace detail
    } // namespace interface
} // namespace alpaka

#endif
