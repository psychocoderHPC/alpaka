/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/CudaHipCommon.hpp"
#include "alpaka/core/Decay.hpp"
#include "alpaka/core/UniformCudaHip.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/math/Complex.hpp"
#include "alpaka/math/TraitsDef.hpp"

#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::math
{
    //! The CUDA built in abs.
    class AbsUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAbs, AbsUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in acos.
    class AcosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAcos, AcosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in acosh.
    class AcoshUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAcosh, AcoshUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in arg.
    class ArgUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathArg, ArgUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in asin.
    class AsinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsin, AsinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in asinh.
    class AsinhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsinh, AsinhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan.
    class AtanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan, AtanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atanh.
    class AtanhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtanh, AtanhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan2.
    class Atan2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cbrt.
    class CbrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCbrt, CbrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in ceil.
    class CeilUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCeil, CeilUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in conj.
    class ConjUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathConj, ConjUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in copysign.
    class CopysignUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathCopysign, CopysignUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cos.
    class CosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCos, CosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cosh.
    class CoshUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCosh, CoshUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in erf.
    class ErfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathErf, ErfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in exp.
    class ExpUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathExp, ExpUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in floor.
    class FloorUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFloor, FloorUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in fma.
    class FmaUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFma, FmaUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in fmod.
    class FmodUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFmod, FmodUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isfinite.
    class IsfiniteUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathIsfinite, IsfiniteUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isinf.
    class IsinfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsinf, IsinfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isnan.
    class IsnanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsnan, IsnanUniformCudaHipBuiltIn>
    {
    };

    // ! The CUDA built in log.
    class LogUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog, LogUniformCudaHipBuiltIn>
    {
    };

    // ! The CUDA built in log2.
    class Log2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog2, Log2UniformCudaHipBuiltIn>
    {
    };

    // ! The CUDA built in log10.
    class Log10UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog10, Log10UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in max.
    class MaxUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in min.
    class MinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMin, MinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in pow.
    class PowUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathPow, PowUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in remainder.
    class RemainderUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathRemainder, RemainderUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA round.
    class RoundUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRound, RoundUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA rsqrt.
    class RsqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sin.
    class SinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSin, SinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sinh.
    class SinhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinh, SinhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sincos.
    class SinCosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sqrt.
    class SqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSqrt, SqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA tan.
    class TanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTan, TanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA tanh.
    class TanhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTanh, TanhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA trunc.
    class TruncUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTrunc, TruncUniformCudaHipBuiltIn>
    {
    };

    //! The standard library math trait specializations.
    class MathUniformCudaHipBuiltIn
        : public AbsUniformCudaHipBuiltIn
        , public AcosUniformCudaHipBuiltIn
        , public AcoshUniformCudaHipBuiltIn
        , public ArgUniformCudaHipBuiltIn
        , public AsinUniformCudaHipBuiltIn
        , public AsinhUniformCudaHipBuiltIn
        , public AtanUniformCudaHipBuiltIn
        , public AtanhUniformCudaHipBuiltIn
        , public Atan2UniformCudaHipBuiltIn
        , public CbrtUniformCudaHipBuiltIn
        , public CeilUniformCudaHipBuiltIn
        , public ConjUniformCudaHipBuiltIn
        , public CopysignUniformCudaHipBuiltIn
        , public CosUniformCudaHipBuiltIn
        , public CoshUniformCudaHipBuiltIn
        , public ErfUniformCudaHipBuiltIn
        , public ExpUniformCudaHipBuiltIn
        , public FloorUniformCudaHipBuiltIn
        , public FmaUniformCudaHipBuiltIn
        , public FmodUniformCudaHipBuiltIn
        , public LogUniformCudaHipBuiltIn
        , public Log2UniformCudaHipBuiltIn
        , public Log10UniformCudaHipBuiltIn
        , public MaxUniformCudaHipBuiltIn
        , public MinUniformCudaHipBuiltIn
        , public PowUniformCudaHipBuiltIn
        , public RemainderUniformCudaHipBuiltIn
        , public RoundUniformCudaHipBuiltIn
        , public RsqrtUniformCudaHipBuiltIn
        , public SinUniformCudaHipBuiltIn
        , public SinhUniformCudaHipBuiltIn
        , public SinCosUniformCudaHipBuiltIn
        , public SqrtUniformCudaHipBuiltIn
        , public TanUniformCudaHipBuiltIn
        , public TanhUniformCudaHipBuiltIn
        , public TruncUniformCudaHipBuiltIn
        , public IsnanUniformCudaHipBuiltIn
        , public IsinfUniformCudaHipBuiltIn
        , public IsfiniteUniformCudaHipBuiltIn
    {
    };

} // namespace alpaka::math

#endif
