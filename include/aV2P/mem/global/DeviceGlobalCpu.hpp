/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/mem/buf/cpu/Copy.hpp"
#include "aV2P/mem/global/Traits.hpp"
#include "aV2P/mem/view/ViewPlainPtr.hpp"

#include <type_traits>

// memcpy specialization for device global variables
namespace aV2P
{

    namespace detail
    {
        template<typename T>
        struct DevGlobalTrait<TagCpuSerial, T>
        {
            using Type = detail::DevGlobalImplGeneric<TagCpuSerial, T>;
        };
    } // namespace detail

    template<
        concepts::Tag TTag,
        typename TViewSrc,
        typename TTypeDst,
        typename TQueue,
        typename std::enable_if_t<std::is_same_v<TTag, TagCpuSerial>, int> = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        aV2P::detail::DevGlobalImplGeneric<TTag, TTypeDst>& viewDst,
        TViewSrc const& viewSrc) -> void
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        auto extent = getExtents(viewSrc);
        auto view = aV2P::ViewPlainPtr<DevCpu, Type, aV2P::Dim<decltype(extent)>, aV2P::Idx<decltype(extent)>>(
            reinterpret_cast<Type*>(const_cast<std::remove_const_t<TTypeDst>*>(&viewDst)),
            aV2P::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    template<
        concepts::Tag TTag,
        typename TTypeSrc,
        typename TViewDstFwd,
        typename TQueue,
        typename std::enable_if_t<std::is_same_v<TTag, TagCpuSerial>, int> = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        TViewDstFwd&& viewDst,
        aV2P::detail::DevGlobalImplGeneric<TTag, TTypeSrc>& viewSrc) -> void
    {
        using Type = std::remove_all_extents_t<TTypeSrc>;
        auto extent = getExtents(viewDst);
        auto view = aV2P::ViewPlainPtr<DevCpu, Type, aV2P::Dim<decltype(extent)>, aV2P::Idx<decltype(extent)>>(
            reinterpret_cast<Type*>(&viewSrc),
            aV2P::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), view, extent));
    }

    template<
        concepts::Tag TTag,
        typename TExtent,
        typename TViewSrc,
        typename TTypeDst,
        typename TQueue,
        typename std::enable_if_t<std::is_same_v<TTag, TagCpuSerial>, int> = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        aV2P::detail::DevGlobalImplGeneric<TTag, TTypeDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent const& extent) -> void
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        auto view = aV2P::ViewPlainPtr<DevCpu, Type, aV2P::Dim<TExtent>, aV2P::Idx<TExtent>>(
            reinterpret_cast<Type*>(const_cast<std::remove_const_t<TTypeDst>*>(&viewDst)),
            aV2P::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    template<
        concepts::Tag TTag,
        typename TExtent,
        typename TTypeSrc,
        typename TViewDstFwd,
        typename TQueue,
        typename std::enable_if_t<std::is_same_v<TTag, TagCpuSerial>, int> = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        TViewDstFwd&& viewDst,
        aV2P::detail::DevGlobalImplGeneric<TTag, TTypeSrc>& viewSrc,
        TExtent const& extent) -> void
    {
        using Type = std::remove_all_extents_t<TTypeSrc>;
        auto view = aV2P::ViewPlainPtr<DevCpu, Type, aV2P::Dim<TExtent>, aV2P::Idx<TExtent>>(
            reinterpret_cast<Type*>(&viewSrc),
            aV2P::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), view, extent));
    }
} // namespace aV2P
