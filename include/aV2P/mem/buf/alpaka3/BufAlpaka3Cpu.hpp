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
        friend ConstBufAlpaka3<TDevSpec, TElem, TDim, TIdx>;

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
        constexpr BufAlpaka3& operator=(BufAlpaka3 const& other) = default;

        constexpr BufAlpaka3& operator=(BufAlpaka3&&) = default;

#if 0
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
#endif
        ALPAKA_FN_HOST decltype(auto) data() const
        {
            return alpaka::onHost::data(m_memImpl);
        }

        ALPAKA_FN_HOST decltype(auto) data()
        {
            return alpaka::onHost::data(m_memImpl);
        }

    public:
        BufType m_memImpl;
        DevAlpaka3<TDevSpec> m_dev;
    };

    template<typename TElem, typename TDim, typename TIdx>
    using BufCpu = BufAlpaka3<typename AccCpuSerial<TDim, TIdx>::DevSpec, TElem, TDim, TIdx>;
} // namespace aV2P

namespace alpaka
{
    namespace internal
    {
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
        struct GetApi::Op<aV2P::BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
        {
            inline constexpr auto operator()(auto&& buff) const
            {
                alpaka::unused(buff);
                return alpaka::getApi(buff.getAlpaka3());
            }
        };
    } // namespace internal

    namespace onHost::internal
    {
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
        struct GetExtents::Op<aV2P::BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
        {
            inline constexpr auto operator()(auto&& buff) const
            {
                alpaka::unused(buff);
                return alpaka::onHost::getExtents(buff.getAlpaka3());
            }
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec, typename TElem, typename TDim, typename TIdx>
        struct GetPitches::Op<aV2P::BufAlpaka3<TDevSpec, TElem, TDim, TIdx>>
        {
            inline constexpr auto operator()(auto&& buff) const
            {
                alpaka::unused(buff);
                return alpaka::onHost::getPitches(buff.getAlpaka3());
            }
        };
    } // namespace onHost::internal
} // namespace alpaka

#include "aV2P/mem/buf/alpaka3/Copy.hpp"
#include "aV2P/mem/buf/alpaka3/Fill.hpp"
#include "aV2P/mem/buf/alpaka3/Set.hpp"
#include "aV2P/mem/buf/alpaka3/traits/BufAlpaka3CpuTraits.hpp"
#include "aV2P/mem/buf/alpaka3/traits/ConstBufAlpaka3CpuTraits.hpp"
