/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/Traits.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/mem/buf/alpaka3/ConstBufAlpaka3Cpu.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"

#include <alpaka/alpaka.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace aV2P
{
    template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
    class BufAlpaka3 : public internal::ViewAccessorType<DevAlpaka3<TDevSpec>, BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
    {
    public:
        using BufType = decltype(alpaka::onHost::alloc<TElem>(
            std::declval<DevAlpaka3<TDevSpec>>().getAlpaka3(),
            std::declval<alpaka::Vec<TIdx, TDim::value>>()));

        auto getAlpaka3() const
        {
            return m_memImpl;
        }

        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufAlpaka3(
            DevAlpaka3<TDevSpec> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extents,
            TExtent const& pitches)
            : m_memImpl{BufType(getApi(dev.m_devImpl), pMem, extents, pitches, deleter)}
            , m_dev{dev}
        {
        }

        ALPAKA_FN_HOST BufAlpaka3(BufType const& sharedBuff, DevAlpaka3<TDevSpec> const& dev)
            : m_memImpl{sharedBuff}
            , m_dev{dev}
        {
        }

        BufAlpaka3(BufAlpaka3 const&) = default;
        BufAlpaka3& operator=(BufAlpaka3 const&) = default;

        ALPAKA_FN_HOST decltype(auto) operator[](TIdx i)
        {
            static_assert(BufType::dim() == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return m_memImpl[i];
        }

        ALPAKA_FN_HOST decltype(auto) operator[](TIdx i) const
        {
            static_assert(BufType::dim() == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return m_memImpl[i];
        }

        ALPAKA_FN_HOST decltype(auto) data() const
        {
            return m_memImpl.data();
        }

        ALPAKA_FN_HOST decltype(auto) data()
        {
            return m_memImpl.data();
        }

    public:
        BufType m_memImpl;
        DevAlpaka3<TDevSpec> m_dev;
    };

    namespace trait
    {
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
        struct GetPitchesInBytes<BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST_ACC constexpr auto operator()(BufAlpaka3<TDevSpec, TElem, TDim, TIdx> const& buf) const
            {
                using Alpaka3Buf = BufAlpaka3<TDevSpec, TElem, TDim, TIdx>;
                return Vec<TDim, TIdx>{buf.getAlpaka3().getPitches()};
            }
        };
    } // namespace trait
} // namespace aV2P

#include "aV2P/mem/buf/alpaka3/Copy.hpp"
#include "aV2P/mem/buf/alpaka3/Set.hpp"
#if 0
#    include "aV2P/mem/buf/alpaka3/Fill.hpp"
#endif
#include "aV2P/mem/buf/alpaka3/traits/BufAlpaka3CpuTraits.hpp"
#include "aV2P/mem/buf/alpaka3/traits/ConstBufAlpaka3CpuTraits.hpp"
