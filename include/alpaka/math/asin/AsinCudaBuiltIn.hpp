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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/asin/Traits.hpp>

//#include <boost/core/ignore_unused.hpp>

#include <cuda_runtime.h>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library asin.
        class AsinCudaBuiltIn
        {
        public:
            using AsinBase = AsinCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library asin trait specialization.
            template<
                typename TArg>
            struct Asin<
                AsinCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto asin(
                    AsinCudaBuiltIn const & /*asin*/,
                    TArg const & arg)
                -> decltype(::asin(arg))
                {
                    //boost::ignore_unused(asin);
                    return ::asin(arg);
                }
            };

            template<>
            struct Asin<
                AsinCudaBuiltIn,
                float>
            {
                __device__ static auto asin(
                    AsinCudaBuiltIn const & asin_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(asin_ctx);
                    return ::asinf(arg);
                }
            };
        }
    }
}

#endif
