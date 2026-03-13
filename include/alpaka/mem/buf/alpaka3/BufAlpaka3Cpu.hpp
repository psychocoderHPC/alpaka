/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/alpaka3/ConstBufAlpaka3Cpu.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"

#include <alpaka/alpaka3.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The CPU memory buffer template implementing muting accessors.
    template<typename TElem, typename TDim, typename TIdx>
    class BufAlpaka3Cpu : public internal::ViewAccessorType<DevAlpaka3Cpu, BufAlpaka3Cpu<TElem, TDim, TIdx>>
    {
    public:
        using BufType = decltype(onHost::alloc<TElem>(
            std::declval<typename DevAlpaka3Cpu::DevType>(),
            std::declval<Vec<TIdx, TDim::value>>()));

        auto getAlpaka3() const
        {
            return m_memImpl;
        }

        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufAlpaka3Cpu(
            DevAlpaka3Cpu const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extents,
            TExtent const& pitches)
            : m_memImpl{BufType(getApi(dev.m_devImpl), pMem, extents, pitches, deleter)}
        {
        }

        ALPAKA_FN_HOST BufAlpaka3Cpu(BufType const& sharedBuff) : m_memImpl{sharedBuff}
        {
        }

        BufAlpaka3Cpu(BufAlpaka3Cpu const&) = default;
        BufAlpaka3Cpu& operator=(BufAlpaka3Cpu const&) = default;

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

    public:
        BufType m_memImpl;
    };
} // namespace alpaka

#include "alpaka/mem/buf/alpaka3/Copy.hpp"
#if 0
#    include "alpaka/mem/buf/alpaka3/Fill.hpp"
#    include "alpaka/mem/buf/alpaka3/Set.hpp"
#endif
#include "alpaka/mem/buf/alpaka3/traits/BufAlpaka3CpuTraits.hpp"
#include "alpaka/mem/buf/alpaka3/traits/ConstBufAlpaka3CpuTraits.hpp"
