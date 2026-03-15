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
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct ConstBufType<DevAlpaka3<TDevSpec>, TElem, TDim, TIdx>
    {
        using type = ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
    };

    //! The ConstBufAlpaka3Cpu device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct DevType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = DevAlpaka3<TDevSpec>;
    };

    //! The ConstBufAlpaka3Cpu device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct GetDev<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> DevAlpaka3<TDevSpec>
        {
            return buf.m_dev;
        }
    };

    //! The ConstBufAlpaka3Cpu dimension getter trait.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct DimType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The ConstBufAlpaka3Cpu memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct ElemType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        // const qualify the element type of the inner view
        using type = TElem const;
    };

    //! The ConstBufAlpaka3Cpu width get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct GetExtents<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
        {
            return buf.m_spBufImpl->m_extentElements;
        }
    };

    //! The ConstBufAlpaka3Cpu native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct GetPtrNative<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return buf.m_spBufImpl->m_pMem;
        }
    };

    //! The ConstBufAlpaka3Cpu pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct GetPtrDev<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>, DevAlpaka3<TDevSpec>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf,
            DevAlpaka3<TDevSpec> const& dev) -> TElem const*
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
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct GetOffsets<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const&) const -> VecOld<TDim, TIdx>
        {
            return VecOld<TDim, TIdx>::zeros();
        }
    };

    //! The ConstBufAlpaka3Cpu idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct IdxType<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The MakeConstBuf trait for constant CPU buffers.
    template<typename TElem, typename TDim, typename TIdx, onHost::concepts::DeviceSpec TDevSpec>
    struct MakeConstBuf<ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf)
            -> ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return buf;
        }

        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>&& buf)
            -> ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>
        {
            return buf;
        }
    };
} // namespace alpaka::trait
