/* Copyright 2023 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/acc/AccAlpaka3Cpu.hpp"
#include "aV2P/rand/RandDefault.hpp"
#include "aV2P/rand/Traits.hpp"

namespace aV2P
{
    namespace rand::distribution::trait
    {
        //! The GPU device random number float normal distribution get trait specialization.
        template<typename T, alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct CreateNormalReal<TAlpakaAcc, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            ALPAKA_FN_HOST_ACC static auto createNormalReal(TAlpakaAcc const& acc) -> gpu::NormalReal<TAlpakaAcc, T>
            {
                return {acc};
            }
        };

        //! The GPU device random number float uniform distribution get trait specialization.
        template<typename T, alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct CreateUniformReal<TAlpakaAcc, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            ALPAKA_FN_HOST_ACC static auto createUniformReal(TAlpakaAcc const& /* acc */) -> gpu::UniformReal<T>
            {
                return {};
            }
        };

        //! The GPU device random number integer uniform distribution get trait specialization.
        template<typename T, alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct CreateUniformUint<TAlpakaAcc, T, std::enable_if_t<std::is_integral_v<T>>>
        {
            ALPAKA_FN_HOST_ACC static auto createUniformUint(TAlpakaAcc const& /* acc */) -> gpu::UniformUint<T>
            {
                return {};
            }
        };
    } // namespace rand::distribution::trait

    namespace rand::engine::trait
    {
        //! The GPU device random number default generator get trait specialization.
        template<alpaka::onAcc::concepts::Acc TAlpakaAcc>
        struct CreateDefault<TAlpakaAcc>
        {
            template<typename TAcc>
            ALPAKA_FN_HOST_ACC static auto createDefault(
                TAcc const& /* acc */,
                std::uint32_t const& seed,
                std::uint32_t const& subsequence,
                std::uint32_t const& offset) -> Philox4x32x10
            {
                return {seed, subsequence, offset};
            }
        };
    } // namespace rand::engine::trait
} // namespace aV2P
