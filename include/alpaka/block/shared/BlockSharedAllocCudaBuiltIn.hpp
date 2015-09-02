/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz, Rene Widera
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

#include <alpaka/block/shared/Traits.hpp>   // AllocVar, AllocArr

#include <alpaka/core/Common.hpp>           // ALPAKA_FN_ACC_CUDA_ONLY

#include <type_traits>                      // std::is_trivially_default_constructible, std::is_trivially_destructible

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            //#############################################################################
            //! The GPU CUDA block shared memory allocator.
            //#############################################################################
            class BlockSharedAllocCudaBuiltIn
            {
            public:
                using BlockSharedAllocBase = BlockSharedAllocCudaBuiltIn;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY BlockSharedAllocCudaBuiltIn() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY BlockSharedAllocCudaBuiltIn(BlockSharedAllocCudaBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY BlockSharedAllocCudaBuiltIn(BlockSharedAllocCudaBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY auto operator=(BlockSharedAllocCudaBuiltIn const &) -> BlockSharedAllocCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY auto operator=(BlockSharedAllocCudaBuiltIn &&) -> BlockSharedAllocCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY /*virtual*/ ~BlockSharedAllocCudaBuiltIn() = default;
            };

            namespace traits
            {
               //#############################################################################
                //!
                //#############################################################################
                template<
                    typename T>
                struct AllocVar<
                    T,
                    BlockSharedAllocCudaBuiltIn>
                {
                    // NOTE: CUDA requires the constructor and destructor of shared objects to be empty.
                    //  That means it is either std::is_trivially_default_constructible or of the form Type(){}.
                    //static_assert(
                    //    std::is_trivially_default_constructible<T>::value,
                    //    "The type of the object to allocate in block shared memory has to be trivially default constructible!");
                    //static_assert(
                    //    std::is_trivially_destructible<T>::value,
                    //    "The type of the object to allocate in block shared memory has to be trivially destructible!");

                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY static auto allocVar(
                        block::shared::BlockSharedAllocCudaBuiltIn const &)
                    -> T &
                    {
                        __shared__ T shMem;
                        return *((T*)(&shMem));
                    }
                };
                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename T,
                    std::size_t TnumElements>
                struct AllocArr<
                    T,
                    TnumElements,
                    BlockSharedAllocCudaBuiltIn>
                {
                    // NOTE: CUDA requires the constructor and destructor of shared objects to be empty.
                    //  That means it is either std::is_trivially_default_constructible or of the form Type(){}.
                    //static_assert(
                    //    std::is_trivially_default_constructible<T>::value,
                    //    "The type of the objects to allocate in block shared memory has to be trivially default constructible!");
                    //static_assert(
                    //    std::is_trivially_destructible<T>::value,
                    //    "The type of the objects to allocate in block shared memory has to be trivially destructible!");
                    static_assert(
                        TnumElements > 0,
                        "The number of elements to allocate in block shared memory must not be zero!");

                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY static auto allocArr(
                        block::shared::BlockSharedAllocCudaBuiltIn const &)
                    -> T *
                    {
                        __shared__ T shMem[TnumElements];
                        return shMem;
                    }
                };
                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct FreeMem<
                    BlockSharedAllocCudaBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY static auto freeMem(
                        block::shared::BlockSharedAllocCudaBuiltIn const &)
                    -> void
                    {
                        // Nothing to do. CUDA block shared memory is automatically freed when all threads left the block.
                    }
                };
            }
        }
    }
}
