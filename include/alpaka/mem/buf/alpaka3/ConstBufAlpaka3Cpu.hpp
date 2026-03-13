/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Vectorize.hpp"
#include "alpaka/dev/DevAlpaka3Cpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/alpaka3/BufAlpaka3Cpu.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/platform/PlatformCpu.hpp"

#include <alpaka/alpaka3.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    // Predeclaration of BufAlpaka3Cpu
    template<typename TElem, typename TDim, typename TIdx>
    class BufAlpaka3Cpu;

    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class ConstBufAlpaka3Cpu : public internal::ViewAccessorType<DevAlpaka3Cpu, ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
    public:
        using BufType = onHost::SharedBuffer<api::Host, TElem, Vec<TIdx, TDim::value>>;

        auto getAlpaka3() const
        {
            return m_memImpl;
        }

        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST ConstBufAlpaka3Cpu(
            DevAlpaka3Cpu const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extents,
            TExtent const& pitches)
            : m_memImpl{BufType(getApi(dev.m_devImpl), pMem, extents, pitches, deleter)}
        {
        }

        ALPAKA_FN_HOST ConstBufAlpaka3Cpu(BufType const& sharedBuff) : m_memImpl{sharedBuff}
        {
        }

        ALPAKA_FN_HOST ConstBufAlpaka3Cpu(BufAlpaka3Cpu<TElem, TDim, TIdx> const& buf) : m_memImpl{buf.m_spBufImpl}
        {
        }

        ALPAKA_FN_HOST ConstBufAlpaka3Cpu(BufAlpaka3Cpu<TElem, TDim, TIdx>&& buf)
            : m_memImpl{std::move(buf.m_spBufImpl)}
        {
        }

    private:
        BufType m_memImpl;

        friend alpaka::trait::GetDev<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetExtents<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrNative<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrDev<ConstBufAlpaka3Cpu<TElem, TDim, TIdx>, DevAlpaka3Cpu>;
    };
} // namespace alpaka
