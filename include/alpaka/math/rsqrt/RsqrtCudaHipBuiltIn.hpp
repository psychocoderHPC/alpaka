 /* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
 
#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #include <cuda_runtime.h>
    #if !BOOST_LANG_CUDA
        #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
    #endif
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)

    #if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
        #include <cuda_runtime_api.h>
    #else
        #if BOOST_COMP_HCC
            #include <math_functions.h>
        #else
            #include <math_functions.hpp>
        #endif
    #endif
    
    #if !BOOST_LANG_HIP
        #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
    #endif
#endif

#include <alpaka/core/Unused.hpp>

#include <type_traits>

#include <alpaka/math/rsqrt/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA rsqrt.
        class RsqrtCudaHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtCudaHipBuiltIn>
        {};

        namespace traits
        {
            //#############################################################################
            //! The CUDA rsqrt trait specialization.
            template<
                typename TArg>
            struct Rsqrt<
                RsqrtCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                __device__ static auto rsqrt(
                    RsqrtCudaHipBuiltIn const & rsqrt_ctx,
                    TArg const & arg)
                -> decltype(::rsqrt(arg))
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrt(arg);
                }
            };
            //! The CUDA rsqrt float specialization.
            template<>
            struct Rsqrt<
                RsqrtCudaHipBuiltIn,
                float>
            {
                __device__ static auto rsqrt(
                    RsqrtCudaHipBuiltIn const & rsqrt_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrtf(arg);
                }
            };
        }
    }
}

#endif