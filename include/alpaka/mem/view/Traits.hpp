/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/dev/Traits.hpp>        // dev::traits::DevType, ...
#include <alpaka/dim/Traits.hpp>        // dim::DimType
#include <alpaka/extent/Traits.hpp>     // extent::GetExtent
#include <alpaka/offset/Traits.hpp>     // offset::GetOffset

#include <alpaka/core/Fold.hpp>         // core::foldr
#include <alpaka/core/Common.hpp>       // ALPAKA_FN_HOST

#include <iosfwd>                       // std::ostream

namespace alpaka
{
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The buffer specifics.
        //-----------------------------------------------------------------------------
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The memory buffer view type trait.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize,
                    typename TSfinae = void>
                struct ViewType;

                //#############################################################################
                //! The memory element type trait.
                //#############################################################################
                template<
                    typename TView,
                    typename TSfinae = void>
                struct ElemType;

                //#############################################################################
                //! The native pointer get trait.
                //#############################################################################
                template<
                    typename TBuf,
                    typename TSfinae = void>
                struct GetPtrNative;

                //#############################################################################
                //! The pointer on device get trait.
                //#############################################################################
                template<
                    typename TBuf,
                    typename TDev,
                    typename TSfinae = void>
                struct GetPtrDev;

                //#############################################################################
                //! The pitch in bytes.
                //! This is the distance in bytes in the linear memory between two consecutive elements in the next higher dimension (TIdx+1).
                //!
                //! The default implementation uses the extent to calculate the pitch.
                //#############################################################################
                template<
                    typename TIdx,
                    typename TView,
                    typename TSfinae = void>
                struct GetPitchBytes
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        TView const & view)
                    -> size::Size<TView>
                    {
                        using IdxSequence = alpaka::core::detail::make_integer_sequence_offset<std::size_t, TIdx::value, dim::Dim<TView>::value - TIdx::value>;
                        return
                            extentsProd(view, IdxSequence())
                            * sizeof(typename ElemType<TView>::type);
                    }
                private:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        std::size_t... TIndices>
                    ALPAKA_FN_HOST static auto extentsProd(
                        TView const & view,
                        alpaka::core::detail::integer_sequence<std::size_t, TIndices...> const &)
                    -> size::Size<TView>
                    {
                        // For the case that the sequence is empty (index out of range), 1 is returned.
                        return
                            core::foldr(
                                std::multiplies<size::Size<TView>>(),
                                1u,
                                extent::getExtent<TIndices>(view)...);
                    }
                };

                //#############################################################################
                //! The memory set trait.
                //!
                //! Fills the buffer with data.
                //#############################################################################
                template<
                    typename TDim,
                    typename TDev,
                    typename TSfinae = void>
                struct Set;

                //#############################################################################
                //! The memory copy trait.
                //!
                //! Copies memory from one buffer into another buffer possibly on a different device.
                //#############################################################################
                template<
                    typename TDim,
                    typename TDevDst,
                    typename TDevSrc,
                    typename TSfinae = void>
                struct Copy;

                //#############################################################################
                //! The memory buffer view creation type trait.
                //#############################################################################
                template<
                    typename TView,
                    typename TSfinae = void>
                struct CreateView;

                //#############################################################################
                //! The buffer trait.
                //#############################################################################
                template<
                    typename TView,
                    typename TSfinae = void>
                struct GetBuf;
            }

            //#############################################################################
            //! The memory element type trait alias template to remove the ::type.
            //#############################################################################
            template<
                typename TView>
            using Elem = typename std::remove_volatile<typename traits::ElemType<TView>::type>::type;

            //#############################################################################
            //! The memory buffer view type trait alias template to remove the ::type.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            using View = typename traits::ViewType<TDev, TElem, TDim, TSize>::type;

            //-----------------------------------------------------------------------------
            //! Gets the native pointer of the memory buffer.
            //!
            //! \param buf The memory buffer.
            //! \return The native pointer.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf>
            ALPAKA_FN_HOST auto getPtrNative(
                TBuf const & buf)
            -> Elem<TBuf> const *
            {
                return traits::GetPtrNative<
                    TBuf>
                ::getPtrNative(
                    buf);
            }
            //-----------------------------------------------------------------------------
            //! Gets the native pointer of the memory buffer.
            //!
            //! \param buf The memory buffer.
            //! \return The native pointer.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf>
            ALPAKA_FN_HOST auto getPtrNative(
                TBuf & buf)
            -> Elem<TBuf> *
            {
                return traits::GetPtrNative<
                    TBuf>
                ::getPtrNative(
                    buf);
            }

            //-----------------------------------------------------------------------------
            //! Gets the pointer to the buffer on the given device.
            //!
            //! \param buf The memory buffer.
            //! \param dev The device.
            //! \return The pointer on the device.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf,
                typename TDev>
            ALPAKA_FN_HOST auto getPtrDev(
                TBuf const & buf,
                TDev const & dev)
            -> Elem<TBuf> const *
            {
                return traits::GetPtrDev<
                    TBuf,
                    TDev>
                ::getPtrDev(
                    buf,
                    dev);
            }
            //-----------------------------------------------------------------------------
            //! Gets the pointer to the buffer on the given device.
            //!
            //! \param buf The memory buffer.
            //! \param dev The device.
            //! \return The pointer on the device.
            //-----------------------------------------------------------------------------
            template<
                typename TBuf,
                typename TDev>
            ALPAKA_FN_HOST auto getPtrDev(
                TBuf & buf,
                TDev const & dev)
            -> Elem<TBuf> *
            {
                return traits::GetPtrDev<
                    TBuf,
                    TDev>
                ::getPtrDev(
                    buf,
                    dev);
            }

            //-----------------------------------------------------------------------------
            //! \return The pitch in bytes. This is the distance between two consecutive rows.
            //-----------------------------------------------------------------------------
            template<
                std::size_t TuiIdx,
                typename TView>
            ALPAKA_FN_HOST auto getPitchBytes(
                TView const & buf)
            -> size::Size<TView>
            {
                return
                    traits::GetPitchBytes<
                        std::integral_constant<std::size_t, TuiIdx>,
                        TView>
                    ::getPitchBytes(
                        buf);
            }

            //-----------------------------------------------------------------------------
            //! Sets the memory to the given value.
            //!
            //! \param buf The memory buffer to fill.
            //! \param byte Value to set for each element of the specified buffer.
            //! \param extents The extents of the buffer to fill.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TExtents>
            ALPAKA_FN_HOST auto set(
                TView & buf,
                std::uint8_t const & byte,
                TExtents const & extents)
            -> void
            {
                static_assert(
                    dim::Dim<TView>::value == dim::Dim<TExtents>::value,
                    "The buffer and the extents are required to have the same dimensionality!");

                traits::Set<
                    dim::Dim<TView>,
                    dev::Dev<TView>>
                ::set(
                    buf,
                    byte,
                    extents);
            }

            //-----------------------------------------------------------------------------
            //! Sets the memory to the given value asynchronously.
            //!
            //! \param buf The memory buffer to fill.
            //! \param byte Value to set for each element of the specified buffer.
            //! \param extents The extents of the buffer to fill.
            //! \param stream The stream to enqueue the buffer fill task into.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TExtents,
                typename TStream>
            ALPAKA_FN_HOST auto set(
                TView & buf,
                std::uint8_t const & byte,
                TExtents const & extents,
                TStream const & stream)
            -> void
            {
                static_assert(
                    dim::Dim<TView>::value == dim::Dim<TExtents>::value,
                    "The buffer and the extents are required to have the same dimensionality!");

                traits::Set<
                    dim::Dim<TView>,
                    dev::Dev<TView>,
                    TStream>
                ::set(
                    buf,
                    byte,
                    extents,
                    stream);
            }

            //-----------------------------------------------------------------------------
            //! Copies memory possibly between different memory spaces.
            //!
            //! \param bufDst The destination memory buffer.
            //! \param bufSrc The source memory buffer.
            //! \param extents The extents of the buffer to copy.
            //-----------------------------------------------------------------------------
            template<
                typename TBufDst,
                typename TBufSrc,
                typename TExtents>
            ALPAKA_FN_HOST auto copy(
                TBufDst & bufDst,
                TBufSrc const & bufSrc,
                TExtents const & extents)
            -> void
            {
                static_assert(
                    dim::Dim<TBufDst>::value == dim::Dim<TBufSrc>::value,
                    "The source and the destination buffers are required to have the same dimensionality!");
                static_assert(
                    dim::Dim<TBufDst>::value == dim::Dim<TExtents>::value,
                    "The destination buffer and the extents are required to have the same dimensionality!");
                static_assert(
                    std::is_same<Elem<TBufDst>, typename std::remove_const<Elem<TBufSrc>>::type>::value,
                    "The source and the destination buffers are required to have the same element type!");

                traits::Copy<
                    dim::Dim<TBufDst>,
                    dev::Dev<TBufDst>,
                    dev::Dev<TBufSrc>>
                ::copy(
                    bufDst,
                    bufSrc,
                    extents);
            }

            //-----------------------------------------------------------------------------
            //! Copies memory possibly between different memory spaces asynchronously.
            //!
            //! \param bufDst The destination memory buffer.
            //! \param bufSrc The source memory buffer.
            //! \param extents The extents of the buffer to copy.
            //! \param stream The stream to enqueue the buffer copy task into.
            //-----------------------------------------------------------------------------
            template<
                typename TBufDst,
                typename TBufSrc,
                typename TExtents,
                typename TStream>
            ALPAKA_FN_HOST auto copy(
                TBufDst & bufDst,
                TBufSrc const & bufSrc,
                TExtents const & extents,
                TStream const & stream)
            -> void
            {
                static_assert(
                    dim::Dim<TBufDst>::value == dim::Dim<TBufSrc>::value,
                    "The source and the destination buffers are required to have the same dimensionality!");
                static_assert(
                    dim::Dim<TBufDst>::value == dim::Dim<TExtents>::value,
                    "The destination buffer and the extents are required to have the same dimensionality!");
                static_assert(
                    std::is_same<Elem<TBufDst>, typename std::remove_const<Elem<TBufSrc>>::type>::value,
                    "The source and the destination buffers are required to have the same element type!");

                traits::Copy<
                    dim::Dim<TBufDst>,
                    dev::Dev<TBufDst>,
                    dev::Dev<TBufSrc>>
                ::copy(
                    bufDst,
                    bufSrc,
                    extents,
                    stream);
            }

            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf>
            ALPAKA_FN_HOST auto createView(
                TBuf const & buf)
            -> decltype(traits::CreateView<TView>::createView(buf))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf);
            }
            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf>
            ALPAKA_FN_HOST auto createView(
                TBuf & buf)
            -> decltype(traits::CreateView<TView>::createView(buf))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf);
            }
            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //! \param extentsElements The extents in elements.
            //! \param relativeOffsetsElements The offsets in elements.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf,
                typename TExtents,
                typename TOffsets>
            ALPAKA_FN_HOST auto createView(
                TBuf const & buf,
                TExtents const & extentsElements,
                TOffsets const & relativeOffsetsElements = TOffsets())
            -> decltype(traits::CreateView<TView>::createView(buf, extentsElements, relativeOffsetsElements))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf,
                    extentsElements,
                    relativeOffsetsElements);
            }
            //-----------------------------------------------------------------------------
            //! Constructor.
            //! \param buf This can be either a memory buffer or a memory view.
            //! \param extentsElements The extents in elements.
            //! \param relativeOffsetsElements The offsets in elements.
            //-----------------------------------------------------------------------------
            template<
                typename TView,
                typename TBuf,
                typename TExtents,
                typename TOffsets>
            ALPAKA_FN_HOST auto createView(
                TBuf & buf,
                TExtents const & extentsElements,
                TOffsets const & relativeOffsetsElements = TOffsets())
            -> decltype(traits::CreateView<TView>::createView(buf, extentsElements, relativeOffsetsElements))
            {
                return traits::CreateView<
                    TView>
                ::createView(
                    buf,
                    extentsElements,
                    relativeOffsetsElements);
            }

            //-----------------------------------------------------------------------------
            //! Gets the memory buffer.
            //!
            //! \param view The object the buffer is received from.
            //! \return The memory buffer.
            //-----------------------------------------------------------------------------
            template<
                typename TView>
            ALPAKA_FN_HOST auto getBuf(
                TView const & view)
            -> decltype(traits::GetBuf<TView>::getBuf(view))
            {
                return traits::GetBuf<
                    TView>
                ::getBuf(
                    view);
            }
            //-----------------------------------------------------------------------------
            //! Gets the memory buffer.
            //!
            //! \param view The object the buffer is received from.
            //! \return The memory buffer.
            //-----------------------------------------------------------------------------
            template<
                typename TView>
            ALPAKA_FN_HOST auto getBuf(
                TView & view)
            -> decltype(traits::GetBuf<TView>::getBuf(view))
            {
                return traits::GetBuf<
                    TView>
                ::getBuf(
                    view);
            }

            namespace detail
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TView>
                struct Print
                {
                    ALPAKA_FN_HOST static auto print(
                        TView const & view,
                        Elem<TView> const * const ptr,
                        Vec<dim::Dim<TView>, size::Size<TView>> const & extents,
                        std::ostream & os,
                        std::string const & elementSeparator,
                        std::string const & rowSeparator,
                        std::string const & rowPrefix,
                        std::string const & rowSuffix)
                    -> void
                    {
                        os << rowPrefix;

                        auto const pitch(view::getPitchBytes<TDim::value+1u>(view));
                        auto const uiLastIdx(extents[TDim::value]-1u);
                        for(auto i(decltype(uiLastIdx)(0)); i<=uiLastIdx ;++i)
                        {
                            Print<
                                dim::DimInt<TDim::value+1u>,
                                TView>
                            ::print(
                                view,
                                reinterpret_cast<Elem<TView> const *>(reinterpret_cast<std::uint8_t const *>(ptr)+i*pitch),
                                extents,
                                os,
                                elementSeparator,
                                rowSeparator,
                                rowPrefix,
                                rowSuffix);

                            // While we are not at the end of a row, add the row separator.
                            if(i != uiLastIdx)
                            {
                                os << rowSeparator;
                            }
                        }

                        os << rowSuffix;
                    }
                };
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                struct Print<
                    dim::DimInt<dim::Dim<TView>::value-1u>,
                    TView>
                {
                    ALPAKA_FN_HOST static auto print(
                        TView const & view,
                        Elem<TView> const * const ptr,
                        Vec<dim::Dim<TView>, size::Size<TView>> const & extents,
                        std::ostream & os,
                        std::string const & elementSeparator,
                        std::string const & rowSeparator,
                        std::string const & rowPrefix,
                        std::string const & rowSuffix)
                    -> void
                    {
                        os << rowPrefix;

                        auto const uiLastIdx(extents[dim::Dim<TView>::value-1u]-1u);
                        for(auto i(decltype(uiLastIdx)(0)); i<=uiLastIdx ;++i)
                        {
                            // Add the current element.
                            os << *(ptr+i);

                            // While we are not at the end of a line, add the element separator.
                            if(i != uiLastIdx)
                            {
                                os << elementSeparator;
                            }
                        }

                        os << rowSuffix;
                    }
                };
            }
            //-----------------------------------------------------------------------------
            //! Prints the content of the view to the given stream.
            // \TODO: Add precision flag.
            // \TODO: Add column alignment flag.
            //-----------------------------------------------------------------------------
            template<
                typename TView>
            ALPAKA_FN_HOST auto print(
                TView const & view,
                std::ostream & os,
                std::string const & elementSeparator = ", ",
                std::string const & rowSeparator = "\n",
                std::string const & rowPrefix = "[",
                std::string const & rowSuffix = "]")
            -> void
            {
                detail::Print<
                    dim::DimInt<0u>,
                    TView>
                ::print(
                    view,
                    view::getPtrNative(view),
                    extent::getExtentsVec(view),
                    os,
                    elementSeparator,
                    rowSeparator,
                    rowPrefix,
                    rowSuffix);
            }
        }
    }
}
