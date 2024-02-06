/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Decay.hpp"
#include "alpaka/math/TraitsDef.hpp"

namespace alpaka::math
{
    //! The standard library abs, implementation covered by the general template.
    class AbsStdLib : public concepts::Implements<ConceptMathAbs, AbsStdLib>
    {
    };

    //! The standard library acos, implementation covered by the general template.
    class AcosStdLib : public concepts::Implements<ConceptMathAcos, AcosStdLib>
    {
    };

    //! The standard library acos, implementation covered by the general template.
    class AcoshStdLib : public concepts::Implements<ConceptMathAcosh, AcoshStdLib>
    {
    };

    //! The standard library arg, implementation covered by the general template.
    class ArgStdLib : public concepts::Implements<ConceptMathArg, ArgStdLib>
    {
    };

    //! The standard library asin, implementation covered by the general template.
    class AsinStdLib : public concepts::Implements<ConceptMathAsin, AsinStdLib>
    {
    };

    //! The standard library asinh, implementation covered by the general template.
    class AsinhStdLib : public concepts::Implements<ConceptMathAsinh, AsinhStdLib>
    {
    };

    //! The standard library atan, implementation covered by the general template.
    class AtanStdLib : public concepts::Implements<ConceptMathAtan, AtanStdLib>
    {
    };

    //! The standard library atanh, implementation covered by the general template.
    class AtanhStdLib : public concepts::Implements<ConceptMathAtanh, AtanhStdLib>
    {
    };

    //! The standard library atan2, implementation covered by the general template.
    class Atan2StdLib : public concepts::Implements<ConceptMathAtan2, Atan2StdLib>
    {
    };

    //! The standard library cbrt, implementation covered by the general template.
    class CbrtStdLib : public concepts::Implements<ConceptMathCbrt, CbrtStdLib>
    {
    };

    //! The standard library ceil, implementation covered by the general template.
    class CeilStdLib : public concepts::Implements<ConceptMathCeil, CeilStdLib>
    {
    };

    //! The standard library conj, implementation covered by the general template.
    class ConjStdLib : public concepts::Implements<ConceptMathConj, ConjStdLib>
    {
    };

    //! The standard library copysign, implementation covered by the general template.
    class CopysignStdLib : public concepts::Implements<ConceptMathCopysign, CopysignStdLib>
    {
    };

    //! The standard library cos, implementation covered by the general template.
    class CosStdLib : public concepts::Implements<ConceptMathCos, CosStdLib>
    {
    };

    //! The standard library cosh, implementation covered by the general template.
    class CoshStdLib : public concepts::Implements<ConceptMathCosh, CoshStdLib>
    {
    };

    //! The standard library erf, implementation covered by the general template.
    class ErfStdLib : public concepts::Implements<ConceptMathErf, ErfStdLib>
    {
    };

    //! The standard library exp, implementation covered by the general template.
    class ExpStdLib : public concepts::Implements<ConceptMathExp, ExpStdLib>
    {
    };

    //! The standard library floor, implementation covered by the general template.
    class FloorStdLib : public concepts::Implements<ConceptMathFloor, FloorStdLib>
    {
    };

    //! The standard library fma, implementation covered by the general template.
    class FmaStdLib : public concepts::Implements<ConceptMathFma, FmaStdLib>
    {
    };

    //! The standard library fmod, implementation covered by the general template.
    class FmodStdLib : public concepts::Implements<ConceptMathFmod, FmodStdLib>
    {
    };

    //! The standard library isfinite, implementation covered by the general template.
    class IsfiniteStdLib : public concepts::Implements<ConceptMathIsfinite, IsfiniteStdLib>
    {
    };

    //! The standard library isinf, implementation covered by the general template.
    class IsinfStdLib : public concepts::Implements<ConceptMathIsinf, IsinfStdLib>
    {
    };

    //! The standard library isnan, implementation covered by the general template.
    class IsnanStdLib : public concepts::Implements<ConceptMathIsnan, IsnanStdLib>
    {
    };

    //! The standard library log, implementation covered by the general template.
    class LogStdLib : public concepts::Implements<ConceptMathLog, LogStdLib>
    {
    };

    //! The standard library log2, implementation covered by the general template.
    class Log2StdLib : public concepts::Implements<ConceptMathLog2, Log2StdLib>
    {
    };

    //! The standard library log10, implementation covered by the general template.
    class Log10StdLib : public concepts::Implements<ConceptMathLog10, Log10StdLib>
    {
    };

    //! The standard library max.
    class MaxStdLib : public concepts::Implements<ConceptMathMax, MaxStdLib>
    {
    };

    //! The standard library min.
    class MinStdLib : public concepts::Implements<ConceptMathMin, MinStdLib>
    {
    };

    //! The standard library pow, implementation covered by the general template.
    class PowStdLib : public concepts::Implements<ConceptMathPow, PowStdLib>
    {
    };

    //! The standard library remainder, implementation covered by the general template.
    class RemainderStdLib : public concepts::Implements<ConceptMathRemainder, RemainderStdLib>
    {
    };

    //! The standard library round, implementation covered by the general template.
    class RoundStdLib : public concepts::Implements<ConceptMathRound, RoundStdLib>
    {
    };

    //! The standard library rsqrt, implementation covered by the general template.
    class RsqrtStdLib : public concepts::Implements<ConceptMathRsqrt, RsqrtStdLib>
    {
    };

    //! The standard library sin, implementation covered by the general template.
    class SinStdLib : public concepts::Implements<ConceptMathSin, SinStdLib>
    {
    };

    //! The standard library sinh, implementation covered by the general template.
    class SinhStdLib : public concepts::Implements<ConceptMathSinh, SinhStdLib>
    {
    };

    //! The standard library sincos, implementation covered by the general template.
    class SinCosStdLib : public concepts::Implements<ConceptMathSinCos, SinCosStdLib>
    {
    };

    //! The standard library sqrt, implementation covered by the general template.
    class SqrtStdLib : public concepts::Implements<ConceptMathSqrt, SqrtStdLib>
    {
    };

    //! The standard library tan, implementation covered by the general template.
    class TanStdLib : public concepts::Implements<ConceptMathTan, TanStdLib>
    {
    };

    //! The standard library tanh, implementation covered by the general template.
    class TanhStdLib : public concepts::Implements<ConceptMathTanh, TanhStdLib>
    {
    };

    //! The standard library trunc, implementation covered by the general template.
    class TruncStdLib : public concepts::Implements<ConceptMathTrunc, TruncStdLib>
    {
    };

    class MathStdLib
        : public AbsStdLib
        , public AcosStdLib
        , public AcoshStdLib
        , public ArgStdLib
        , public AsinStdLib
        , public AsinhStdLib
        , public AtanStdLib
        , public AtanhStdLib
        , public Atan2StdLib
        , public CbrtStdLib
        , public CeilStdLib
        , public ConjStdLib
        , public CopysignStdLib
        , public CosStdLib
        , public CoshStdLib
        , public ErfStdLib
        , public ExpStdLib
        , public FloorStdLib
        , public FmaStdLib
        , public FmodStdLib
        , public LogStdLib
        , public Log2StdLib
        , public Log10StdLib
        , public MaxStdLib
        , public MinStdLib
        , public PowStdLib
        , public RemainderStdLib
        , public RoundStdLib
        , public RsqrtStdLib
        , public SinStdLib
        , public SinhStdLib
        , public SinCosStdLib
        , public SqrtStdLib
        , public TanStdLib
        , public TanhStdLib
        , public TruncStdLib
        , public IsnanStdLib
        , public IsinfStdLib
        , public IsfiniteStdLib
    {
    };
} // namespace alpaka::math
