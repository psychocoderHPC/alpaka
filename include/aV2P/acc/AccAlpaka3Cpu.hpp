/* Copyright 2025 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "aV2P/dev/DevAlpaka3Cpu.hpp"

// Specialized traits.
#include "aV2P/acc/Traits.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/kernel/Traits.hpp"
#include "aV2P/platform/Traits.hpp"

// Implementation details.
#include "aV2P/acc/Tag.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/dev/DevAlpaka3Cpu.hpp"

#ifdef __cpp_lib_format
#    include <format>
#endif
#include <memory>
#include <string>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace aV2P
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
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
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

    template<typename TDim, typename TIdx>
    using AccAlpaka3Cpu = AccAlpaka3<TDim, TIdx, alpaka::api::Host, alpaka::deviceKind::Cpu, alpaka::exec::CpuSerial>;

    template<typename TDim, typename TIdx>
    using AccAlpaka3Omp
        = AccAlpaka3<TDim, TIdx, alpaka::api::Host, alpaka::deviceKind::Cpu, alpaka::exec::CpuOmpBlocks>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagAlpaka3Cpu, TDim, TIdx>
        {
            using type = aV2P::AccAlpaka3Cpu<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = aV2P::TagAlpaka3Cpu;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<aV2P::TagAlpaka3Omp, TDim, TIdx>
        {
            using type = aV2P::AccAlpaka3Omp<TDim, TIdx>;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<aV2P::AccAlpaka3Omp<TDim, TIdx>>
        {
            using type = aV2P::TagAlpaka3Omp;
        };

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
                return {// m_multiProcessorCount
                        prop.multiProcessorCount,
                        // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        prop.maxThreadsPerBlock,
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::ones(),
                        // m_blockThreadCountMax
                        static_cast<TIdx>(1),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
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
            static constexpr auto getIdx(alpaka::onAcc::Acc<TDict> const& acc, TWorkDiv const& workDiv)
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

#endif
