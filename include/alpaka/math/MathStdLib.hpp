/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Decay.hpp"
#include "alpaka/math/Traits.hpp"
#include "alpaka/math/MathStdLibConcept.hpp"

namespace alpaka::math
{
    namespace trait
    {
        //! The standard library max trait specialization.
        template<typename Tx, typename Ty>
        struct Max<MaxStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            ALPAKA_FN_HOST auto operator()(Tx const& x, Ty const& y)
            {
                using std::fmax;
                using std::max;

                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return max(x, y);
                else if constexpr(
                    is_decayed_v<Tx, float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double>
                    || is_decayed_v<Ty, double>)
                    return fmax(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
            }
        };

        //! The standard library min trait specialization.
        template<typename Tx, typename Ty>
        struct Min<MinStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            ALPAKA_FN_HOST auto operator()(Tx const& x, Ty const& y)
            {
                using std::fmin;
                using std::min;

                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return min(x, y);
                else if constexpr(
                    is_decayed_v<Tx, float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double>
                    || is_decayed_v<Ty, double>)
                    return fmin(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
            }
        };
    } // namespace trait

} // namespace alpaka::math

#include "alpaka/math/Traits.hpp"
