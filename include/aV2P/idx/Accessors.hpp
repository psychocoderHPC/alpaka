/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Common.hpp"
#include "aV2P/core/Interface.hpp"
#include "aV2P/core/Positioning.hpp"
#include "aV2P/dim/DimIntegralConst.hpp"
#include "aV2P/dim/Traits.hpp"
#include "aV2P/idx/Traits.hpp"
#include "aV2P/workdiv/Traits.hpp"

#include <utility>

namespace aV2P
{
    //! Get the indices requested.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOrigin, typename TUnit, typename TIdx, typename TWorkDiv>
    ALPAKA_FN_HOST_ACC auto getIdx(TIdx const& idx, TWorkDiv const& workDiv) -> aV2P::Vec<Dim<TWorkDiv>, Idx<TIdx>>
    {
        return trait::GetIdx<TIdx, TOrigin, TUnit>::getIdx(idx, workDiv);
    }

    //! Get the indices requested.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOrigin, typename TUnit, typename TIdxWorkDiv>
    ALPAKA_FN_HOST_ACC auto getIdx(TIdxWorkDiv const& idxWorkDiv) -> aV2P::Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        return trait::GetIdx<TIdxWorkDiv, TOrigin, TUnit>::getIdx(idxWorkDiv, idxWorkDiv);
    }

    namespace trait
    {
        //! The grid block index get trait specialization for classes with IdxGbBase member type.
        template<typename TIdxGb>
        struct GetIdx<TIdxGb, origin::Grid, unit::Blocks>
        {
            using ImplementationBase = interface::ImplementationBase<ConceptIdxGb, TIdxGb>;

            //! \return The index of the current thread in the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TWorkDiv>
            ALPAKA_FN_HOST_ACC static auto getIdx(TIdxGb const& idx, TWorkDiv const& workDiv)
                -> aV2P::Vec<Dim<ImplementationBase>, Idx<ImplementationBase>>
            {
                return trait::GetIdx<ImplementationBase, origin::Grid, unit::Blocks>::getIdx(idx, workDiv);
            }
        };

        //! The block thread index get trait specialization for classes with IdxBtBase member type.
        template<typename TIdxBt>
        struct GetIdx<TIdxBt, origin::Block, unit::Threads>
        {
            using ImplementationBase = interface::ImplementationBase<ConceptIdxBt, TIdxBt>;

            //! \return The index of the current thread in the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TWorkDiv>
            ALPAKA_FN_HOST_ACC static auto getIdx(TIdxBt const& idx, TWorkDiv const& workDiv)
                -> aV2P::Vec<Dim<ImplementationBase>, Idx<ImplementationBase>>
            {
                return trait::GetIdx<ImplementationBase, origin::Block, unit::Threads>::getIdx(idx, workDiv);
            }
        };

        //! The grid thread index get trait specialization.
        template<typename TIdx>
        struct GetIdx<TIdx, origin::Grid, unit::Threads>
        {
            //! \return The index of the current thread in the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TWorkDiv>
            ALPAKA_FN_HOST_ACC static auto getIdx(TIdx const& idx, TWorkDiv const& workDiv)
            {
                return aV2P::getIdx<origin::Grid, unit::Blocks>(idx, workDiv)
                           * getWorkDiv<origin::Block, unit::Threads>(workDiv)
                       + aV2P::getIdx<origin::Block, unit::Threads>(idx, workDiv);
            }
        };
    } // namespace trait

    //! Get the index of the first element this thread computes.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIdxWorkDiv, typename TGridThreadIdx, typename TThreadElemExtent>
    ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(
        [[maybe_unused]] TIdxWorkDiv const& idxWorkDiv,
        TGridThreadIdx const& gridThreadIdx,
        TThreadElemExtent const& threadElemExtent) -> aV2P::Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        return gridThreadIdx * threadElemExtent;
    }

    //! Get the index of the first element this thread computes.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIdxWorkDiv, typename TGridThreadIdx>
    ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(TIdxWorkDiv const& idxWorkDiv, TGridThreadIdx const& gridThreadIdx)
        -> aV2P::Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        auto const threadElemExtent(aV2P::getWorkDiv<aV2P::Thread, aV2P::Elems>(idxWorkDiv));
        return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx, threadElemExtent);
    }

    //! Get the index of the first element this thread computes.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIdxWorkDiv>
    ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(TIdxWorkDiv const& idxWorkDiv)
        -> aV2P::Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        auto const gridThreadIdx(aV2P::getIdx<aV2P::Grid, aV2P::Threads>(idxWorkDiv));
        return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx);
    }
} // namespace aV2P
