/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber,
 * Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/dev/DevAlpaka3Cpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/alpaka3/BufAlpaka3Cpu.hpp"
#include "alpaka/mem/buf/alpaka3/ConstBufAlpaka3Cpu.hpp"

namespace alpaka::trait
{
    //! The CPU device memory const-buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct ConstBufType<DevAlpaka3Cpu, TElem, TDim, TIdx>
    {
        using type = ConstBufAlpaka3Cpu<TElem, TDim, TIdx>;
    };

    //! The ConstBufAlpaka3Cpu device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct DevType<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        using type = DevAlpaka3Cpu;
    };

    //! The ConstBufAlpaka3Cpu device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetDev<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(ConstBufAlpaka3Cpu<TElem, TDim, TIdx> const& buf) -> DevAlpaka3Cpu
        {
            return buf.m_spBufImpl->m_dev;
        }
    };

    //! The ConstBufAlpaka3Cpu dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx>
    struct DimType<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The ConstBufAlpaka3Cpu memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct ElemType<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        // const qualify the element type of the inner view
        using type = TElem const;
    };

    //! The ConstBufAlpaka3Cpu width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetExtents<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufAlpaka3Cpu<TElem, TDim, TIdx> const& buf)
        {
            return buf.m_spBufImpl->m_extentElements;
        }
    };

    //! The ConstBufAlpaka3Cpu native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrNative<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(ConstBufAlpaka3Cpu<TElem, TDim, TIdx> const& buf) -> TElem const*
        {
            return buf.m_spBufImpl->m_pMem;
        }
    };

    //! The ConstBufAlpaka3Cpu pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>, DevAlpaka3Cpu>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            ConstBufAlpaka3Cpu<TElem, TDim, TIdx> const& buf,
            DevAlpaka3Cpu const& dev) -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spBufImpl->m_pMem;
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }
    };

    //! The ConstBufAlpaka3Cpu offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct GetOffsets<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufAlpaka3Cpu<TElem, TDim, TIdx> const&) const -> VecOld<TDim, TIdx>
        {
            return VecOld<TDim, TIdx>::zeros();
        }
    };

    //! The ConstBufAlpaka3Cpu idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx>
    struct IdxType<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for constant CPU buffers.
    template<typename TElem, typename TDim, typename TIdx>
    struct MakeConstBuf<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufAlpaka3Cpu<TElem, TDim, TIdx> const& buf)
            -> ConstBufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            return buf;
        }

        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufAlpaka3Cpu<TElem, TDim, TIdx>&& buf)
            -> ConstBufAlpaka3Cpu<TElem, TDim, TIdx>
        {
            return buf;
        }
    };
} // namespace alpaka::trait
