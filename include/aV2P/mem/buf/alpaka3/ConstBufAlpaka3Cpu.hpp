/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Vectorize.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/alpaka3/BufAlpaka3Cpu.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/meta/DependentFalseType.hpp"
#include "aV2P/platform/PlatformAlpaka3Cpu.hpp"

#include <alpaka/alpaka.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace aV2P
{
    // Predeclaration of BufAlpaka3Cpu
    template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
    class BufAlpaka3;

    //! The CPU memory buffer.
    template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
    class ConstBufAlpaka3
        : public internal::ViewAccessorType<DevAlpaka3<TDevSpec>, ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
    public:
        using BufType = decltype(alpaka::onHost::alloc<TElem>(
            std::declval<typename DevAlpaka3<TDevSpec>::DevType>(),
            std::declval<alpaka::Vec<TIdx, TDim::value>>()));

        auto getAlpaka3() const
        {
            return m_memImpl;
        }

        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST ConstBufAlpaka3(
            DevAlpaka3<TDevSpec> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extents,
            TExtent const& pitches)
            : m_memImpl{BufType(getApi(dev.m_devImpl), pMem, extents, pitches, deleter)}
            , m_dev{dev}
        {
        }

        ALPAKA_FN_HOST ConstBufAlpaka3(BufType const& sharedBuff, DevAlpaka3<TDevSpec> dev)
            : m_memImpl{sharedBuff}
            , m_dev{dev}
        {
        }

        ALPAKA_FN_HOST ConstBufAlpaka3(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf, DevAlpaka3<TDevSpec> dev)
            : m_memImpl{buf.m_spBufImpl}
            , m_dev{dev}
        {
        }

        ALPAKA_FN_HOST ConstBufAlpaka3(BufAlpaka3<TDevSpec, TElem, TDim, TIdx>&& buf, DevAlpaka3<TDevSpec> dev)
            : m_memImpl{std::move(buf.m_spBufImpl)}
            , m_dev{dev}
        {
        }

    private:
        BufType m_memImpl;
        DevAlpaka3<TDevSpec> m_dev;
    };
} // namespace aV2P
