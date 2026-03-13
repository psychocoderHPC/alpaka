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
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelAlpaka3;

    //! The CPU serial accelerator.
    //!
    //! This accelerator allows serial kernel execution on a CPU device.
    //! The block idx is restricted to 1x1x1 and all blocks are executed serially so there is no parallelism at all.
    template<typename TDim, typename TIdx>
    class AccAlpaka3Cpu final : public interface::Implements<InterfaceAcc, AccAlpaka3Cpu<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelAlpaka3;

        AccAlpaka3Cpu(AccAlpaka3Cpu const&) = delete;
        AccAlpaka3Cpu(AccAlpaka3Cpu&&) = delete;
        auto operator=(AccAlpaka3Cpu const&) -> AccAlpaka3Cpu& = delete;
        auto operator=(AccAlpaka3Cpu&&) -> AccAlpaka3Cpu& = delete;
    };

    namespace trait
    {
        //! The CPU serial accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = AccAlpaka3Cpu<TDim, TIdx>;
        };

        //! The CPU serial single thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccAlpaka3Cpu<TDim, TIdx>> : std::true_type
        {
        };

        //! The CPU serial multi thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccAlpaka3Cpu<TDim, TIdx>> : std::false_type
        {
        };

        //! The CPU serial accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccAlpaka3Cpu<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(DevAlpaka3Cpu const& dev) -> AccDevProps<TDim, TIdx>
            {
                return {// m_multiProcessorCount
                        static_cast<TIdx>(1),
                        // m_gridBlockExtentMax
                        VecOld<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        VecOld<TDim, TIdx>::ones(),
                        // m_blockThreadCountMax
                        static_cast<TIdx>(1),
                        // m_threadElemExtentMax
                        VecOld<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(0u),
                        // m_globalMemSizeBytes
                        getMemBytes(dev)};
            }
        };

        //! The CPU serial accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccAlpaka3Cpu<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
#    if ALPAKA_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wexit-time-destructors"
#    endif
                using namespace std::literals;
                static std::string const accName =
#    ifdef __cpp_lib_format
                    std::format("AccAlpaka3<{},{}>", TDim::value, core::demangled<TIdx>);
#    else
                    "AccAlpaka3<"s + std::to_string(TDim::value) + ","s + std::string(core::demangled<TIdx>) + ">"s;
#    endif
                return accName;
#    if ALPAKA_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
            }
        };

        //! The CPU serial accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = DevAlpaka3Cpu;
        };

        //! The CPU serial accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU serial accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccAlpaka3Cpu<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
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
                        + getAccName<AccAlpaka3Cpu<TDim, TIdx>>() + ". Threads per block should be 1!");
                }

                return TaskKernelAlpaka3<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU serial execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PlatformType<AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = PlatformAlpaka3Cpu;
        };

        //! The CPU serial accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccAlpaka3Cpu<TDim, TIdx>>
        {
            using type = alpaka::TagAlpaka3Cpu;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagAlpaka3Cpu, TDim, TIdx>
        {
            using type = alpaka::AccAlpaka3Cpu<TDim, TIdx>;
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
            template<typename TBase>
            static auto implements(Implements<TInterface, TBase>&) -> std::true_type;
            static auto implements(...) -> std::false_type;

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
                template<typename TBase>
                static auto implementer(Implements<TInterface, TBase>&) -> TBase;

                using type = onAcc::Acc<TDict>;
            };
        } // namespace detail
    } // namespace interface
} // namespace alpaka

#endif
