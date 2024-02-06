/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Sergei Bastrakov,
 *                Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

#include <cmath>
#include <complex>
#if __has_include(<version>) // Not part of the C++17 standard but all major standard libraries include this
#    include <version>
#endif
#ifdef __cpp_lib_math_constants
#    include <numbers>
#endif

namespace alpaka::math
{
    namespace constants
    {
#ifdef __cpp_lib_math_constants
        inline constexpr double e = std::numbers::e;
        inline constexpr double log2e = std::numbers::log2e;
        inline constexpr double log10e = std::numbers::log10e;
        inline constexpr double pi = std::numbers::pi;
        inline constexpr double inv_pi = std::numbers::inv_pi;
        inline constexpr double ln2 = std::numbers::ln2;
        inline constexpr double ln10 = std::numbers::ln10;
        inline constexpr double sqrt2 = std::numbers::sqrt2;

        template<typename T>
        inline constexpr T e_v = std::numbers::e_v<T>;

        template<typename T>
        inline constexpr T log2e_v = std::numbers::log2e_v<T>;

        template<typename T>
        inline constexpr T log10e_v = std::numbers::log10e_v<T>;

        template<typename T>
        inline constexpr T pi_v = std::numbers::pi_v<T>;

        template<typename T>
        inline constexpr T inv_pi_v = std::numbers::inv_pi_v<T>;

        template<typename T>
        inline constexpr T ln2_v = std::numbers::ln2_v<T>;

        template<typename T>
        inline constexpr T ln10_v = std::numbers::ln10_v<T>;

        template<typename T>
        inline constexpr T sqrt2_v = std::numbers::sqrt2_v<T>;
#else
        inline constexpr double e = M_E;
        inline constexpr double log2e = M_LOG2E;
        inline constexpr double log10e = M_LOG10E;
        inline constexpr double pi = M_PI;
        inline constexpr double inv_pi = M_1_PI;
        inline constexpr double ln2 = M_LN2;
        inline constexpr double ln10 = M_LN10;
        inline constexpr double sqrt2 = M_SQRT2;

        template<typename T>
        inline constexpr T e_v = static_cast<T>(e);

        template<typename T>
        inline constexpr T log2e_v = static_cast<T>(log2e);

        template<typename T>
        inline constexpr T log10e_v = static_cast<T>(log10e);

        template<typename T>
        inline constexpr T pi_v = static_cast<T>(pi);

        template<typename T>
        inline constexpr T inv_pi_v = static_cast<T>(inv_pi);

        template<typename T>
        inline constexpr T ln2_v = static_cast<T>(ln2);

        template<typename T>
        inline constexpr T ln10_v = static_cast<T>(ln10);

        template<typename T>
        inline constexpr T sqrt2_v = static_cast<T>(sqrt2);

        // Use predefined float constants when available
#    if defined(M_Ef)
        template<>
        inline constexpr float e_v<float> = M_Ef;
#    endif

#    if defined(M_LOG2Ef)
        template<>
        inline constexpr float log2e_v<float> = M_LOG2Ef;
#    endif

#    if defined(M_LOG10Ef)
        template<>
        inline constexpr float log10e_v<float> = M_LOG10Ef;
#    endif

#    if defined(M_PIf)
        template<>
        inline constexpr float pi_v<float> = M_PIf;
#    endif

#    if defined(M_1_PIf)
        template<>
        inline constexpr float inv_pi_v<float> = M_1_PIf;
#    endif

#    if defined(M_LN2f)
        template<>
        inline constexpr float ln2_v<float> = M_LN2f;
#    endif

#    if defined(M_LN10f)
        template<>
        inline constexpr float ln10_v<float> = M_LN10f;
#    endif

#    if defined(M_SQRT2f)
        template<>
        inline constexpr float sqrt2_v<float> = M_SQRT2f;
#    endif

#endif
    } // namespace constants

    struct ConceptMathAbs
    {
    };

    struct ConceptMathAcos
    {
    };

    struct ConceptMathAcosh
    {
    };

    struct ConceptMathArg
    {
    };

    struct ConceptMathAsin
    {
    };

    struct ConceptMathAsinh
    {
    };

    struct ConceptMathAtan
    {
    };

    struct ConceptMathAtanh
    {
    };

    struct ConceptMathAtan2
    {
    };

    struct ConceptMathCbrt
    {
    };

    struct ConceptMathCeil
    {
    };

    struct ConceptMathConj
    {
    };

    struct ConceptMathCopysign
    {
    };

    struct ConceptMathCos
    {
    };

    struct ConceptMathCosh
    {
    };

    struct ConceptMathErf
    {
    };

    struct ConceptMathExp
    {
    };

    struct ConceptMathFloor
    {
    };

    struct ConceptMathFma
    {
    };

    struct ConceptMathFmod
    {
    };

    struct ConceptMathIsfinite
    {
    };

    struct ConceptMathIsinf
    {
    };

    struct ConceptMathIsnan
    {
    };

    struct ConceptMathLog
    {
    };

    struct ConceptMathLog2
    {
    };

    struct ConceptMathLog10
    {
    };

    struct ConceptMathMax
    {
    };

    struct ConceptMathMin
    {
    };

    struct ConceptMathPow
    {
    };

    struct ConceptMathRemainder
    {
    };

    struct ConceptMathRound
    {
    };

    struct ConceptMathRsqrt
    {
    };

    struct ConceptMathSin
    {
    };

    struct ConceptMathSinh
    {
    };

    struct ConceptMathSinCos
    {
    };

    struct ConceptMathSqrt
    {
    };

    struct ConceptMathTan
    {
    };

    struct ConceptMathTanh
    {
    };

    struct ConceptMathTrunc
    {
    };

    //! The math traits.
    namespace trait
    {
        //! The abs trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Abs
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find abs(TArg) in the namespace of your type.
                using std::abs;
                return abs(arg);
            }
        };

        //! The acos trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Acos
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find acos(TArg) in the namespace of your type.
                using std::acos;
                return acos(arg);
            }
        };

        //! The acosh trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Acosh
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find acosh(TArg) in the namespace of your type.
                using std::acosh;
                return acosh(arg);
            }
        };

        //! The arg trait.
        template<typename ConceptBase, typename TArgument, typename TSfinae = void>
        struct Arg
        {
            // It is unclear why this is needed here and not in other math trait structs. But removing it causes
            // warnings with calling a __host__ function from a __host__ __device__ function when building for CUDA.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto operator()( TArgument const& argument)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find arg(TArgument) in the namespace of your type.
                using std::arg;
                return arg(argument);
            }
        };

        //! The asin trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Asin
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find asin(TArg) in the namespace of your type.
                using std::asin;
                return asin(arg);
            }
        };

        //! The asin trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Asinh
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find asin(TArg) in the namespace of your type.
                using std::asinh;
                return asinh(arg);
            }
        };

        //! The atan trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Atan
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find atan(TArg) in the namespace of your type.
                using std::atan;
                return atan(arg);
            }
        };

        //! The atanh trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Atanh
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find atanh(TArg) in the namespace of your type.
                using std::atanh;
                return atanh(arg);
            }
        };

        //! The atan2 trait.
        template<typename ConceptBase, typename Ty, typename Tx, typename TSfinae = void>
        struct Atan2
        {
            ALPAKA_FN_HOST_ACC auto operator()( Ty const& y, Tx const& x)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find atan2(Tx, Ty) in the namespace of your type.
                using std::atan2;
                return atan2(y, x);
            }
        };

        //! The cbrt trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Cbrt
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find cbrt(TArg) in the namespace of your type.
                using std::cbrt;
                return cbrt(arg);
            } //! The erf trait.
        };

        //! The ceil trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Ceil
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find ceil(TArg) in the namespace of your type.
                using std::ceil;
                return ceil(arg);
            }
        };

        //! The conj trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Conj
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find conj(TArg) in the namespace of your type.
                using std::conj;
                return conj(arg);
            }
        };

        //! The copysign trait.
        template<typename ConceptBase, typename TMag, typename TSgn, typename TSfinae = void>
        struct Copysign
        {
            ALPAKA_FN_HOST_ACC auto operator()( TMag const& mag, TSgn const& sgn)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find copysign(TMag, TSgn) in the namespace of your type.
                using std::copysign;
                return copysign(mag, sgn);
            }
        };

        //! The cos trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Cos
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find cos(TArg) in the namespace of your type.
                using std::cos;
                return cos(arg);
            }
        };

        //! The cosh trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Cosh
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find cos(TArg) in the namespace of your type.
                using std::cosh;
                return cosh(arg);
            }
        };

        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Erf
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find erf(TArg) in the namespace of your type.
                using std::erf;
                return erf(arg);
            }
        };

        //! The exp trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Exp
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find exp(TArg) in the namespace of your type.
                using std::exp;
                return exp(arg);
            }
        };

        //! The floor trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Floor
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find floor(TArg) in the namespace of your type.
                using std::floor;
                return floor(arg);
            }
        };

        //! The fma trait.
        template<typename ConceptBase, typename Tx, typename Ty, typename Tz, typename TSfinae = void>
        struct Fma
        {
            ALPAKA_FN_HOST_ACC auto operator()( Tx const& x, Ty const& y, Tz const& z)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find fma(Tx, Ty, Tz) in the namespace of your type.
                using std::fma;
                return fma(x, y, z);
            }
        };

        //! The fmod trait.
        template<typename ConceptBase, typename Tx, typename Ty, typename TSfinae = void>
        struct Fmod
        {
            ALPAKA_FN_HOST_ACC auto operator()( Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find fmod(Tx, Ty) in the namespace of your type.
                using std::fmod;
                return fmod(x, y);
            }
        };

        //! The isfinite trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Isfinite
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find isfinite(TArg) in the namespace of your type.
                using std::isfinite;
                return isfinite(arg);
            }
        };

        //! The isinf trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Isinf
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find isinf(TArg) in the namespace of your type.
                using std::isinf;
                return isinf(arg);
            }
        };

        //! The isnan trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Isnan
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find isnan(TArg) in the namespace of your type.
                using std::isnan;
                return isnan(arg);
            }
        };

        //! The log trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Log
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find log(TArg) in the namespace of your type.
                using std::log;
                return log(arg);
            }
        };

        //! The bas 2 log trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Log2
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find log2(TArg) in the namespace of your type.
                using std::log2;
                return log2(arg);
            }
        };

        //! The base 10 log trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Log10
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find log10(TArg) in the namespace of your type.
                using std::log10;
                return log10(arg);
            }
        };

        //! The max trait.
        template<typename ConceptBase, typename Tx, typename Ty, typename TSfinae = void>
        struct Max
        {
            ALPAKA_FN_HOST_ACC auto operator()( Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find max(Tx, Ty) in the namespace of your type.
                using std::max;
                return max(x, y);
            }
        };

        //! The min trait.
        template<typename ConceptBase, typename Tx, typename Ty, typename TSfinae = void>
        struct Min
        {
            ALPAKA_FN_HOST_ACC auto operator()( Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find min(Tx, Ty) in the namespace of your type.
                using std::min;
                return min(x, y);
            }
        };

        //! The pow trait.
        template<typename ConceptBase, typename TBase, typename TExp, typename TSfinae = void>
        struct Pow
        {
            ALPAKA_FN_HOST_ACC auto operator()( TBase const& base, TExp const& exp)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find pow(base, exp) in the namespace of your type.
                using std::pow;
                return pow(base, exp);
            }
        };

        //! The remainder trait.
        template<typename ConceptBase, typename Tx, typename Ty, typename TSfinae = void>
        struct Remainder
        {
            ALPAKA_FN_HOST_ACC auto operator()( Tx const& x, Ty const& y)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find remainder(Tx, Ty) in the namespace of your type.
                using std::remainder;
                return remainder(x, y);
            }
        };

        //! The round trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Round
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find round(TArg) in the namespace of your type.
                using std::round;
                return round(arg);
            }
        };

        //! The round trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Lround
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find lround(TArg) in the namespace of your type.
                using std::lround;
                return lround(arg);
            }
        };

        //! The round trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Llround
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find llround(TArg) in the namespace of your type.
                using std::llround;
                return llround(arg);
            }
        };

        namespace detail
        {
            //! Fallback implementation when no better ADL match was found
            template<typename TArg>
            ALPAKA_FN_HOST_ACC auto rsqrt(TArg const& arg)
            {
                // Still use ADL to try find sqrt(arg)
                using std::sqrt;
                return static_cast<TArg>(1) / sqrt(arg);
            }
        } // namespace detail

        //! The rsqrt trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Rsqrt
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find rsqrt(TArg) in the namespace of your type.
                using detail::rsqrt;
                return rsqrt(arg);
            }
        };

        //! The sin trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Sin
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find sin(TArg) in the namespace of your type.
                using std::sin;
                return sin(arg);
            }
        };

        //! The sin trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Sinh
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find sin(TArg) in the namespace of your type.
                using std::sinh;
                return sinh(arg);
            }
        };

        namespace detail
        {
            //! Fallback implementation when no better ADL match was found
            template<typename TArg>
            ALPAKA_FN_HOST_ACC auto sincos(TArg const& arg, TArg& result_sin, TArg& result_cos)
            {
                // Still use ADL to try find sin(arg) and cos(arg)
                using std::sin;
                result_sin = sin(arg);
                using std::cos;
                result_cos = cos(arg);
            }
        } // namespace detail

        //! The sincos trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct SinCos
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg, TArg& result_sin, TArg& result_cos)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find sincos(TArg, TArg&, TArg&) in the namespace of your type.
                using detail::sincos;
                return sincos(arg, result_sin, result_cos);
            }
        };

        //! The sqrt trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Sqrt
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find sqrt(TArg) in the namespace of your type.
                using std::sqrt;
                return sqrt(arg);
            }
        };

        //! The tan trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Tan
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find tan(TArg) in the namespace of your type.
                using std::tan;
                return tan(arg);
            }
        };

        //! The tanh trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Tanh
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find tanh(TArg) in the namespace of your type.
                using std::tanh;
                return tanh(arg);
            }
        };

        //! The trunc trait.
        template<typename ConceptBase, typename TArg, typename TSfinae = void>
        struct Trunc
        {
            ALPAKA_FN_HOST_ACC auto operator()( TArg const& arg)
            {
                // This is an ADL call. If you get a compile error here then your type is not supported by the
                // backend and we could not find trunc(TArg) in the namespace of your type.
                using std::trunc;
                return trunc(arg);
            }
        };
    } // namespace trait

} // namespace alpaka::math
