/**
* \file
* Copyright 2014-2015 Benjarsqrt Worpitz
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

#include <alpaka/math/rsqrt/Traits.hpp>  // Rsqrt

//#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_arithmetic
#include <math_functions.hpp>           // ::rsqrt

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library rsqrt.
        //#############################################################################
        class RsqrtCudaBuiltIn
        {
        public:
            using RsqrtBase = RsqrtCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library rsqrt trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Rsqrt<
                RsqrtCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto rsqrt(
                    RsqrtCudaBuiltIn const & rsqrt,
                    TArg const & arg)
                -> decltype(::rsqrt(arg))
                {
                    //boost::ignore_unused(rsqrt);
                    return ::rsqrt(arg);
                }
            };
        }
    }
}
