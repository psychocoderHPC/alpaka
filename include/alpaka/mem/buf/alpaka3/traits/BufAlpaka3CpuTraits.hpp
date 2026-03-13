/* Copyright 2025 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber,
 *                Anton Reinhard, Maria Michailidi
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/dev/DevAlpaka3Cpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/alpaka3/BufAlpaka3Cpu.hpp"

namespace alpaka::trait
{
    //! The CPU device memory buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct BufType<DevAlpaka3Cpu, TElem, TDim, TIdx>
    {
        using type = BufAlpaka3Cpu<TElem, TDim, TIdx>;
    };

    //!  The BufAlpaka3Cpu device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct DevType<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        using type = DevAlpaka3Cpu;
    };

    //! The BufAlpaka3Cpu device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetDev<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(BufAlpaka3Cpu<TElem, TDim, TIdx> const& buf) -> DevAlpaka3Cpu
        {
            return buf.m_spBufImpl->m_dev;
        }
    };

    //! The BufAlpaka3Cpu dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx>
    struct DimType<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The BufAlpaka3Cpu memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct ElemType<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The BufAlpaka3Cpu width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetExtents<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST VecOld<TDim, TIdx> operator()(BufAlpaka3Cpu<TElem, TDim, TIdx> const& buf)
        {
            return VecOld<TDim, TIdx>{buf.getAlpaka3().getExtents()};
        }
    };

    //! The BufAlpaka3Cpu native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrNative<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(BufAlpaka3Cpu<TElem, TDim, TIdx> const& buf) -> TElem const*
        {
            return buf.getAlpaka3().data();
        }

        ALPAKA_FN_HOST static auto getPtrNative(BufAlpaka3Cpu<TElem, TDim, TIdx>& buf) -> TElem*
        {
            return buf.getAlpaka3().data();
        }
    };

    //! The BufAlpaka3Cpu pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<BufAlpaka3Cpu<TElem, TDim, TIdx>, DevAlpaka3Cpu>
    {
        ALPAKA_FN_HOST static auto getPtrDev(BufAlpaka3Cpu<TElem, TDim, TIdx> const& buf, DevAlpaka3Cpu const& dev)
            -> TElem const*
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

        ALPAKA_FN_HOST static auto getPtrDev(BufAlpaka3Cpu<TElem, TDim, TIdx>& buf, DevAlpaka3Cpu const& dev) -> TElem*
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
    template<typename TElem, typename TDim, typename TIdx>
    struct GetOffsets<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufAlpaka3Cpu<TElem, TDim, TIdx> const& /*buf*/) const -> VecOld<TDim, TIdx>
        {
            return VecOld<TDim, TIdx>::zeros();
        }
    };

    //! The BufAlpaka3Cpu idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct IdxType<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for CPU buffers.
    template<typename TElem, typename TDim, typename TIdx>
    struct MakeConstBuf<BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(BufAlpaka3Cpu<TElem, TDim, TIdx> const& buf)
            -> ConstBufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            return ConstBufAlpaka3Cpu<TElem, TDim, TIdx>(buf);
        }

        ALPAKA_FN_HOST static auto makeConstBuf(BufAlpaka3Cpu<TElem, TDim, TIdx>&& buf)
            -> ConstBufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            return ConstBufAlpaka3Cpu<TElem, TDim, TIdx>(std::move(buf));
        }
    };

    //! The BufAlpaka3Cpu memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct BufAlloc<TElem, TDim, TIdx, DevAlpaka3Cpu>
    {
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocBuf(DevAlpaka3Cpu const& dev, TExtent const& extent)
            -> BufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            if constexpr(std::is_integral_v<TExtent>)
                return onHost::alloc<TElem>(dev.getAlpaka3(), Vec{extent});
            else
                return onHost::alloc<TElem>(dev.getAlpaka3(), extent.getAlpaka3());
        }
    };

    //! The BufAlpaka3Cpu stream-ordered memory allocation capability trait specialization.
    template<typename TDim>
    struct HasAsyncBufSupport<TDim, DevAlpaka3Cpu> : public std::true_type
    {
    };

    //! The BufAlpaka3Cpu stream-ordered memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct AsyncBufAlloc<TElem, TDim, TIdx, DevAlpaka3Cpu>
    {
        template<typename TQueue, typename TExtent>
        ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent)
            -> BufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            return {onHost::allocDeferred<TElem>(queue.getAlpaka3(), extent.getAlpaka3())};
        }
    };

    //! The pinned/mapped memory allocation capability trait specialization.
    template<>
    struct HasMappedBufSupport<PlatformAlpaka3Cpu> : public std::true_type
    {
    };

    //! The pinned/mapped memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped<PlatformAlpaka3Cpu, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocMappedBuf(
            DevAlpaka3Cpu const& dev,
            PlatformAlpaka3Cpu const& /*platform*/,
            TExtent const& extent) -> BufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            return {onHost::allocMapped<TElem>(dev.getAlpaka3(), extent.getAlpaka3())};
        }
    };

    //! The unified/managed memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct BufAllocManaged<PlatformAlpaka3Cpu, TElem, TDim, TIdx>
    {
        template<typename TExtent>
        ALPAKA_FN_HOST static auto allocManagedBuf(
            DevAlpaka3Cpu const& dev,
            PlatformAlpaka3Cpu const& /*platform*/,
            TExtent const& extent) -> BufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            return {onHost::allocUnified<TElem>(dev.getAlpaka3(), extent.getAlpaka3())};
        }
    };
} // namespace alpaka::trait
