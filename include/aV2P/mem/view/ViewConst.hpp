/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/dev/Traits.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/extent/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/mem/view/Traits.hpp"
#include "aV2P/mem/view/ViewAccessOps.hpp"
#include "aV2P/offset/Traits.hpp"

namespace aV2P
{
    //! A non-modifiable wrapper around a view. This view acts as the wrapped view, but the underlying data is only
    //! exposed const-qualified.
    template<typename TView>
    struct ViewConst : internal::ViewAccessorType<aV2P::Dev<TView>, ViewConst<TView>>
    {
        using Dev = aV2P::Dev<TView>;

        static_assert(!std::is_const_v<TView>, "ViewConst must be instantiated with a non-const type");
        static_assert(
            !std::is_reference_v<TView>,
            "This is not implemented"); // It might even be dangerous for ViewConst to store a reference to the wrapped
                                        // view, as this decouples the wrapped view's lifetime.

        ALPAKA_FN_HOST ViewConst(TView const& view) : m_view(view)
        {
        }

        ALPAKA_FN_HOST ViewConst(TView&& view) : m_view(std::move(view))
        {
        }

        TView m_view;
    };

    template<typename TView>
    ViewConst(TView) -> ViewConst<std::decay_t<TView>>;

    namespace trait
    {
        template<typename TView>
        struct DevType<ViewConst<TView>> : DevType<TView>
        {
        };

        template<typename TView>
        struct GetDev<ViewConst<TView>>
        {
            ALPAKA_FN_HOST static auto getDev(ViewConst<TView> const& view)
            {
                return aV2P::getDev(view.m_view);
            }
        };

        template<typename TView>
        struct DimType<ViewConst<TView>> : DimType<TView>
        {
        };

        template<typename TView>
        struct ElemType<ViewConst<TView>>
        {
            // const qualify the element type of the inner view
            using type = typename ElemType<TView>::type const;
        };

        template<typename TView>
        struct GetExtents<ViewConst<TView>>
        {
            ALPAKA_FN_HOST auto operator()(ViewConst<TView> const& view) const
            {
                return getExtents(view.m_view);
            }
        };

        template<typename TView>
        struct GetPtrNative<ViewConst<TView>>
        {
            using TElem = typename ElemType<TView>::type;

            // const qualify the element type of the inner view
            ALPAKA_FN_HOST static auto getPtrNative(ViewConst<TView> const& view) -> TElem const*
            {
                return aV2P::getPtrNative(view.m_view);
            }
        };

        template<typename TView>
        struct GetPitchesInBytes<ViewConst<TView>>
        {
            ALPAKA_FN_HOST auto operator()(ViewConst<TView> const& view) const
            {
                return aV2P::getPitchesInBytes(view.m_view);
            }
        };

        template<typename TView>
        struct GetOffsets<ViewConst<TView>>
        {
            ALPAKA_FN_HOST auto operator()(ViewConst<TView> const& view) const
            {
                return aV2P::getOffsets(view.m_view);
            }
        };

        template<typename TView>
        struct IdxType<ViewConst<TView>> : IdxType<TView>
        {
        };
    } // namespace trait
} // namespace aV2P

namespace alpaka
{
    namespace internal
    {
        template<typename TView>
        struct GetApi::Op<aV2P::ViewConst<TView>>
        {
            inline constexpr auto operator()(auto&& buff) const
            {
                alpaka::unused(buff);
                return alpaka::getApi(typename ALPAKA_TYPEOF(buff)::Dev::DevSpec{});
            }
        };
    } // namespace internal

    namespace onHost::internal
    {
        template<typename TView>
        struct GetExtents::Op<aV2P::ViewConst<TView>>
        {
            inline constexpr auto operator()(auto&& buff) const
            {
                alpaka::unused(buff);
                return aV2P::getExtents(buff).getAlpaka3();
            }
        };

        template<typename TView>
        struct GetPitches::Op<aV2P::ViewConst<TView>>
        {
            inline constexpr auto operator()(auto&& buff) const
            {
                alpaka::unused(buff);
                return aV2P::getPitchesInBytes(buff).getAlpaka3();
            }
        };
    } // namespace onHost::internal
} // namespace alpaka
