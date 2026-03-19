/* Copyright 2025 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber,
 *                Anton Reinhard, Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "aV2P/dev/DevAlpaka3Cpu.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/alpaka3/BufAlpaka3Cpu.hpp"

namespace aV2P::trait
{
    //! The CPU device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct BufType<DevAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        using type = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
    };

    //!  The BufAlpaka3Cpu device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct DevType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = DevAlpaka3<TDevSpec>;
    };

    //! The BufAlpaka3Cpu device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct GetDev<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) -> DevAlpaka3<TDevSpec>
        {
            return buf.m_dev;
        }
    };

    //! The BufAlpaka3Cpu dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct DimType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The BufAlpaka3Cpu memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct ElemType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
    struct GetPitchesInBytes<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST_ACC constexpr auto operator()(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) const
        {
            return Vec<TDim, TIdx>{buf.getAlpaka3().getPitches()};
        }
    };

    //! The BufAlpaka3Cpu width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct GetExtents<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST Vec<TDim, TIdx> operator()(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
        {
            return Vec<TDim, TIdx>{buf.getAlpaka3().getExtents()};
        }
    };

    //! The BufAlpaka3Cpu native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct GetPtrNative<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) -> TElem const*
        {
            return buf.getAlpaka3().data();
        }

        ALPAKA_FN_HOST static auto getPtrNative(BufAlpaka3<TDevSpec, TElem, TDim, TIdx>& buf) -> TElem*
        {
            return buf.getAlpaka3().data();
        }
    };

    //! The BufAlpaka3Cpu pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct GetPtrDev<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>, DevAlpaka3<TDevSpec>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf,
            DevAlpaka3<TDevSpec> const& dev) -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return buf.getAlpaka3().data();
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            BufAlpaka3<TDevSpec, TElem, TDim, TIdx>& buf,
            DevAlpaka3<TDevSpec> const& dev) -> TElem*
        {
            if(dev == getDev(buf))
            {
                return buf.getAlpaka3().data();
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }
    };

    //! The BufAlpaka3Cpu offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct GetOffsets<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& /*buf*/) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The BufAlpaka3Cpu idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct IdxType<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for CPU buffers.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct MakeConstBuf<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>(buf);
        }

        ALPAKA_FN_HOST static auto makeConstBuf(BufAlpaka3<TDevSpec, TElem, TDim, TIdx>&& buf)
            -> ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>(std::move(buf));
        }
    };

    //! The BufAlpaka3Cpu memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct BufAlloc<TElem, TDim, TIdx, DevAlpaka3<TDevSpec>>
    {
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocBuf(DevAlpaka3<TDevSpec> const& dev, TExtent const& extent)
            -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            using Buff = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
            if constexpr(std::is_integral_v<TExtent>)
                return Buff{alpaka::onHost::alloc<TElem>(dev.getAlpaka3(), alpaka::Vec{extent}), dev};
            else
                return Buff{alpaka::onHost::alloc<TElem>(dev.getAlpaka3(), extent.getAlpaka3()), dev};
        }
    };

    //! The BufAlpaka3Cpu stream-ordered memory allocation capability trait specialization.
    template<typename TDim, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct HasAsyncBufSupport<TDim, DevAlpaka3<TDevSpec>> : public std::true_type
    {
    };

    //! The BufAlpaka3Cpu stream-ordered memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct AsyncBufAlloc<TElem, TDim, TIdx, DevAlpaka3<TDevSpec>>
    {
        template<typename TQueue, typename TExtent>
        ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent)
            -> BufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            using Buff = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
            if constexpr(std::is_integral_v<TExtent>)
                return Buff{
                    alpaka::onHost::allocDeferred<TElem>(queue.getAlpaka3(), alpaka::Vec{extent}),
                    aV2P::getDev(queue)};
            else
                return Buff{
                    alpaka::onHost::allocDeferred<TElem>(queue.getAlpaka3(), extent.getAlpaka3()),
                    aV2P::getDev(queue)};
        }
    };

    //! The pinned/mapped memory allocation capability trait specialization.
    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct HasMappedBufSupport<PlatformAlpaka3<TDevSpec>> : public std::true_type
    {
    };

    //! The pinned/mapped memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct BufAllocMapped<PlatformAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        /**
         *  @return Buffer with CPU signature and not like in alpaka3 with GPU device kind
         */
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocMappedBuf(
            DevCpu const& dev,
            PlatformAlpaka3<TDevSpec> const& platform,
            TExtent const& extent) -> BufAlpaka3<typename DevCpu::DevSpec, TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            using CpuDevSpec = typename DevCpu::DevSpec;
            using Buff = BufAlpaka3<CpuDevSpec, TElem, TDim, TIdx>;
            using AlpakaBuffType = typename Buff::BufType;
            /* alpaka3 requires a device where the host memory should be available, we use the first device as dummy.
             * In all implementations the memory should be visible on all devices.
             */
            auto alpaka3PlatformDev = platform.getAlpaka3().makeDevice(0);
            if constexpr(std::is_integral_v<TExtent>)
            {
                auto alpaka3Buf = alpaka::onHost::allocMapped<TElem>(alpaka3PlatformDev, alpaka::Vec{extent});
                /* @attention we force reinterpret the alpaka3 device buffer into a host buffer, that very, very, ...
                 * dangerous but works because the deleter in the buffer is a void function. Don't do it at home!!
                 */
                auto buffer = *reinterpret_cast<AlpakaBuffType*>(&alpaka3Buf);
                return Buff{buffer, dev};
            }
            else
            {
                auto alpaka3Buf = alpaka::onHost::allocMapped<TElem>(alpaka3PlatformDev, extent.getAlpaka3());
                /* @attention we force reinterpret the alpaka3 device buffer into a host buffer, that very, very, ...
                 * dangerous but works because the deleter in the buffer is a void function. Don't do it at home!!
                 */
                auto buffer = *reinterpret_cast<AlpakaBuffType*>(&alpaka3Buf);
                return Buff{buffer, dev};
            }
        }
    };

    //! The unified/managed memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct BufAllocManaged<PlatformAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        /**
         *  @return Buffer with CPU signature and not like in alpaka3 with GPU device kind
         */
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocManagedBuf(
            DevCpu const& dev,
            PlatformAlpaka3<TDevSpec> const& platform,
            TExtent const& extent) -> BufAlpaka3<typename DevCpu::DevSpec, TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            using CpuDevSpec = typename DevCpu::DevSpec;
            using Buff = BufAlpaka3<CpuDevSpec, TElem, TDim, TIdx>;
            using AlpakaBuffType = typename Buff::BufType;
            /* alpaka3 requires a device where the host memory should be available, we use the first device as dummy.
             * In all implementations the memory should be visible on all devices.
             */
            auto alpaka3PlatformDev = platform.getAlpaka3().makeDevice(0);
            if constexpr(std::is_integral_v<TExtent>)
            {
                auto alpaka3Buf = alpaka::onHost::allocUnified<TElem>(alpaka3PlatformDev, alpaka::Vec{extent});
                /* @attention we force reinterpret the alpaka3 device buffer into a host buffer, that very, very, ...
                 * dangerous but works because the deleter in the buffer is a void function. Don't do it at home!!
                 */
                auto buffer = *reinterpret_cast<AlpakaBuffType*>(&alpaka3Buf);
                return Buff{buffer, dev};
            }
            else
            {
                auto alpaka3Buf = alpaka::onHost::allocUnified<TElem>(alpaka3PlatformDev, extent.getAlpaka3());
                /* @attention we force reinterpret the alpaka3 device buffer into a host buffer, that very, very, ...
                 * dangerous but works because the deleter in the buffer is a void function. Don't do it at home!!
                 */
                auto buffer = *reinterpret_cast<AlpakaBuffType*>(&alpaka3Buf);
                return Buff{buffer, dev};
            }
        }
    };
} // namespace aV2P::trait
