/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Sergei Bastrakov,
 *                Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/math/MathStdLibConcept.hpp"
#include "alpaka/math/MathUniformCudaHipConcept.hpp"

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
    struct ConceptMath
    {
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1 && defined(__HIP__))
        using type = MathUniformCudaHipBuiltIn;
#elif defined(__SYCL_DEVICE_ONLY__)
        using type = MathGenericSycl;
#else
        using type = MathStdLib;
#endif
    };

    //! Computes the absolute value.
    //!
    //! \tparam T The type of the object specializing Abs.
    //! \tparam TArg The arg type.
    //! \param abs_ctx The object specializing Abs.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto abs(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAbs, MathBase>;
        return trait::Abs<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the principal value of the arc cosine.
    //!
    //! The valid real argument range is [-1.0, 1.0]. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.
    //!
    //! \tparam TArg The arg type.
    //! \param acos_ctx The object specializing Acos.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto acos(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAcos, MathBase>;
        return trait::Acos<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the principal value of the hyperbolic arc cosine.
    //!
    //! The valid real argument range is [1.0, Inf]. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.
    //!
    //! \tparam TArg The arg type.
    //! \param acosh_ctx The object specializing Acos.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto acosh(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAcosh, MathBase>;
        return trait::Acosh<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the complex argument of the value.
    //!
    //! \tparam T The type of the object specializing Arg.
    //! \tparam TArgument The argument type.
    //! \param arg_ctx The object specializing Arg.
    //! \param argument The argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArgument>
    ALPAKA_FN_HOST_ACC auto arg(TArgument const& argument)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathArg, MathBase>;
        return trait::Arg<ImplementationBase, TArgument>{}(argument);
    }

    //! Computes the principal value of the arc sine.
    //!
    //! The valid real argument range is [-1.0, 1.0]. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.
    //!
    //! \tparam TArg The arg type.
    //! \param asin_ctx The object specializing Asin.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto asin(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAsin, MathBase>;
        return trait::Asin<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the principal value of the hyperbolic arc sine.
    //!
    //! \tparam TArg The arg type.
    //! \param asinh_ctx The object specializing Asin.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto asinh(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAsinh, MathBase>;
        return trait::Asinh<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the principal value of the arc tangent.
    //!
    //! \tparam TArg The arg type.
    //! \param atan_ctx The object specializing Atan.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto atan(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAtan, MathBase>;
        return trait::Atan<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the principal value of the hyperbolic arc tangent.
    //!
    //! The valid real argument range is [-1.0, 1.0]. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.

    //! \tparam TArg The arg type.
    //! \param atanh_ctx The object specializing Atanh.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto atanh(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAtanh, MathBase>;
        return trait::Atanh<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the arc tangent of y/x using the signs of arguments to determine the correct quadrant.
    //!
    //! \tparam T The type of the object specializing Atan2.
    //! \tparam Ty The y arg type.
    //! \tparam Tx The x arg type.
    //! \param atan2_ctx The object specializing Atan2.
    //! \param y The y arg.
    //! \param x The x arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Ty, typename Tx>
    ALPAKA_FN_HOST_ACC auto atan2(Ty const& y, Tx const& x)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathAtan2, MathBase>;
        return trait::Atan2<ImplementationBase, Ty, Tx>{}(y, x);
    }

    //! Computes the cbrt.
    //!
    //! \tparam T The type of the object specializing Cbrt.
    //! \tparam TArg The arg type.
    //! \param cbrt_ctx The object specializing Cbrt.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto cbrt(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCbrt, MathBase>;
        return trait::Cbrt<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the smallest integer value not less than arg.
    //!
    //! \tparam T The type of the object specializing Ceil.
    //! \tparam TArg The arg type.
    //! \param ceil_ctx The object specializing Ceil.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto ceil(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCeil, MathBase>;
        return trait::Ceil<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the complex conjugate of arg.
    //!
    //! \tparam T The type of the object specializing Conj.
    //! \tparam TArg The arg type.
    //! \param conj_ctx The object specializing Conj.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto conj(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathConj, MathBase>;
        return trait::Conj<ImplementationBase, TArg>{}(arg);
    }

    //! Creates a value with the magnitude of mag and the sign of sgn.
    //!
    //! \tparam T The type of the object specializing Copysign.
    //! \tparam TMag The mag type.
    //! \tparam TSgn The sgn type.
    //! \param copysign_ctx The object specializing Copysign.
    //! \param mag The mag.
    //! \param sgn The sgn.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TMag, typename TSgn>
    ALPAKA_FN_HOST_ACC auto copysign(TMag const& mag, TSgn const& sgn)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCopysign, MathBase>;
        return trait::Copysign<ImplementationBase, TMag, TSgn>{}(mag, sgn);
    }

    //! Computes the cosine (measured in radians).
    //!
    //! \tparam T The type of the object specializing Cos.
    //! \tparam TArg The arg type.
    //! \param cos_ctx The object specializing Cos.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto cos(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCos, MathBase>;
        return trait::Cos<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the hyperbolic cosine (measured in radians).
    //!
    //! \tparam T The type of the object specializing Cos.
    //! \tparam TArg The arg type.
    //! \param cosh_ctx The object specializing Cos.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto cosh(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathCosh, MathBase>;
        return trait::Cosh<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the error function of arg.
    //!
    //! \tparam T The type of the object specializing Erf.
    //! \tparam TArg The arg type.
    //! \param erf_ctx The object specializing Erf.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto erf(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathErf, MathBase>;
        return trait::Erf<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the e (Euler's number, 2.7182818) raised to the given power arg.
    //!
    //! \tparam T The type of the object specializing Exp.
    //! \tparam TArg The arg type.
    //! \param exp_ctx The object specializing Exp.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto exp(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathExp, MathBase>;
        return trait::Exp<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the largest integer value not greater than arg.
    //!
    //! \tparam T The type of the object specializing Floor.
    //! \tparam TArg The arg type.
    //! \param floor_ctx The object specializing Floor.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto floor(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathFloor, MathBase>;
        return trait::Floor<ImplementationBase, TArg>{}(arg);
    }

    //! Computes x * y + z as if to infinite precision and rounded only once to fit the result type.
    //!
    //! \tparam T The type of the object specializing Fma.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \tparam Tz The type of the third argument.
    //! \param fma_ctx The object specializing .
    //! \param x The first argument.
    //! \param y The second argument.
    //! \param z The third argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Tx, typename Ty, typename Tz>
    ALPAKA_FN_HOST_ACC auto fma(Tx const& x, Ty const& y, Tz const& z)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathFma, MathBase>;
        return trait::Fma<ImplementationBase, Tx, Ty, Tz>{}(x, y, z);
    }

    //! Computes the floating-point remainder of the division operation x/y.
    //!
    //! \tparam T The type of the object specializing Fmod.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param fmod_ctx The object specializing Fmod.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto fmod(Tx const& x, Ty const& y)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathFmod, MathBase>;
        return trait::Fmod<ImplementationBase, Tx, Ty>{}(x, y);
    }

    //! Checks if given value is finite.
    //!
    //! \tparam T The type of the object specializing Isfinite.
    //! \tparam TArg The arg type.
    //! \param ctx The object specializing Isfinite.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto isfinite(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathIsfinite, MathBase>;
        return trait::Isfinite<ImplementationBase, TArg>{}(arg);
    }

    //! Checks if given value is inf.
    //!
    //! \tparam T The type of the object specializing Isinf.
    //! \tparam TArg The arg type.
    //! \param ctx The object specializing Isinf.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto isinf(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathIsinf, MathBase>;
        return trait::Isinf<ImplementationBase, TArg>{}(arg);
    }

    //! Checks if given value is NaN.
    //!
    //! \tparam T The type of the object specializing Isnan.
    //! \tparam TArg The arg type.
    //! \param ctx The object specializing Isnan.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto isnan(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathIsnan, MathBase>;
        return trait::Isnan<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the the natural (base e) logarithm of arg.
    //!
    //! Valid real arguments are non-negative. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Log.
    //! \tparam TArg The arg type.
    //! \param log_ctx The object specializing Log.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto log(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathLog, MathBase>;
        return trait::Log<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the the natural (base 2) logarithm of arg.
    //!
    //! Valid real arguments are non-negative. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Log2.
    //! \tparam TArg The arg type.
    //! \param log2_ctx The object specializing Log2.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto log2(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathLog2, MathBase>;
        return trait::Log2<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the the natural (base 10) logarithm of arg.
    //!
    //! Valid real arguments are non-negative. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Log10.
    //! \tparam TArg The arg type.
    //! \param log10_ctx The object specializing Log10.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto log10(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathLog10, MathBase>;
        return trait::Log10<ImplementationBase, TArg>{}(arg);
    }

    //! Returns the larger of two arguments.
    //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
    //!
    //! \tparam T The type of the object specializing Max.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param max_ctx The object specializing Max.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto max(Tx const& x, Ty const& y)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathMax, MathBase>;
        return trait::Max<ImplementationBase, Tx, Ty>{}(x, y);
    }

    //! Returns the smaller of two arguments.
    //! NaNs are treated as missing data (between a NaN and a numeric value, the numeric value is chosen).
    //!
    //! \tparam T The type of the object specializing Min.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param min_ctx The object specializing Min.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto min(Tx const& x, Ty const& y)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathMin, MathBase>;
        return trait::Min<ImplementationBase, Tx, Ty>{}(x, y);
    }

    //! Computes the value of base raised to the power exp.
    //!
    //! Valid real arguments for base are non-negative. For other values
    //! the result may depend on the backend and compilation options, will
    //! likely be NaN.
    //!
    //! \tparam T The type of the object specializing Pow.
    //! \tparam TBase The base type.
    //! \tparam TExp The exponent type.
    //! \param pow_ctx The object specializing Pow.
    //! \param base The base.
    //! \param exp The exponent.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TBase, typename TExp>
    ALPAKA_FN_HOST_ACC auto pow(TBase const& base, TExp const& exp)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathPow, MathBase>;
        return trait::Pow<ImplementationBase, TBase, TExp>{}(base, exp);
    }

    //! Computes the IEEE remainder of the floating point division operation x/y.
    //!
    //! \tparam T The type of the object specializing Remainder.
    //! \tparam Tx The type of the first argument.
    //! \tparam Ty The type of the second argument.
    //! \param remainder_ctx The object specializing Max.
    //! \param x The first argument.
    //! \param y The second argument.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Tx, typename Ty>
    ALPAKA_FN_HOST_ACC auto remainder(Tx const& x, Ty const& y)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRemainder, MathBase>;
        return trait::Remainder<ImplementationBase, Tx, Ty>{}(x, y);
    }

    //! Computes the nearest integer value to arg (in floating-point format), rounding halfway cases away from
    //! zero, regardless of the current rounding mode.
    //!
    //! \tparam T The type of the object specializing Round.
    //! \tparam TArg The arg type.
    //! \param round_ctx The object specializing Round.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto round(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, MathBase>;
        return trait::Round<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
    //! regardless of the current rounding mode.
    //!
    //! \tparam T The type of the object specializing Round.
    //! \tparam TArg The arg type.
    //! \param lround_ctx The object specializing Round.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto lround(TArg const& arg) -> long int
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, MathBase>;
        return trait::Lround<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the nearest integer value to arg (in integer format), rounding halfway cases away from zero,
    //! regardless of the current rounding mode.
    //!
    //! \tparam T The type of the object specializing Round.
    //! \tparam TArg The arg type.
    //! \param llround_ctx The object specializing Round.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto llround(TArg const& arg) -> long long int
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRound, MathBase>;
        return trait::Llround<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the rsqrt.
    //!
    //! Valid real arguments are positive. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Rsqrt.
    //! \tparam TArg The arg type.
    //! \param rsqrt_ctx The object specializing Rsqrt.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto rsqrt(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathRsqrt, MathBase>;
        return trait::Rsqrt<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the sine (measured in radians).
    //!
    //! \tparam T The type of the object specializing Sin.
    //! \tparam TArg The arg type.
    //! \param sin_ctx The object specializing Sin.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto sin(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathSin, MathBase>;
        return trait::Sin<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the hyperbolic sine (measured in radians).
    //!
    //! \tparam T The type of the object specializing Sin.
    //! \tparam TArg The arg type.
    //! \param sinh_ctx The object specializing Sin.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto sinh(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathSinh, MathBase>;
        return trait::Sinh<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the sine and cosine (measured in radians).
    //!
    //! \tparam T The type of the object specializing SinCos.
    //! \tparam TArg The arg type.
    //! \param sincos_ctx The object specializing SinCos.
    //! \param arg The arg.
    //! \param result_sin result of sine
    //! \param result_cos result of cosine
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto sincos(TArg const& arg, TArg& result_sin, TArg& result_cos) -> void
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathSinCos, MathBase>;
        trait::SinCos<ImplementationBase, TArg>{}(arg, result_sin, result_cos);
    }

    //! Computes the square root of arg.
    //!
    //! Valid real arguments are non-negative. For other values the result
    //! may depend on the backend and compilation options, will likely
    //! be NaN.
    //!
    //! \tparam T The type of the object specializing Sqrt.
    //! \tparam TArg The arg type.
    //! \param sqrt_ctx The object specializing Sqrt.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto sqrt(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathSqrt, MathBase>;
        return trait::Sqrt<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the tangent (measured in radians).
    //!
    //! \tparam T The type of the object specializing Tan.
    //! \tparam TArg The arg type.
    //! \param tan_ctx The object specializing Tan.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto tan(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathTan, MathBase>;
        return trait::Tan<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the hyperbolic tangent (measured in radians).
    //!
    //! \tparam T The type of the object specializing Tanh.
    //! \tparam TArg The arg type.
    //! \param tanh_ctx The object specializing Tanh.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto tanh(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathTanh, MathBase>;
        return trait::Tanh<ImplementationBase, TArg>{}(arg);
    }

    //! Computes the nearest integer not greater in magnitude than arg.
    //!
    //! \tparam T The type of the object specializing Trunc.
    //! \tparam TArg The arg type.
    //! \param trunc_ctx The object specializing Trunc.
    //! \param arg The arg.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TArg>
    ALPAKA_FN_HOST_ACC auto trunc(TArg const& arg)
    {
        using MathBase = typename ConceptMath::type;
        using ImplementationBase = concepts::ImplementationBase<ConceptMathTrunc, MathBase>;
        return trait::Trunc<ImplementationBase, TArg>{}(arg);
    }
} // namespace alpaka::math

#include "alpaka/math/MathUniformCudaHipBuiltIn.hpp"
#include "alpaka/math/MathStdLib.hpp"