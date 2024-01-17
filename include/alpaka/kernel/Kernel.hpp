/* Copyright 2022 Benjamin Worpitz, Bert Wesarg, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Specialized traits.
#include "alpaka/core/Common.hpp"
#include "alpaka/platform/Traits.hpp"

#include <alpaka/core/RemoveRestrict.hpp>

#include <tuple>
#include <type_traits>

namespace alpaka
{
    template<typename TAcc, typename TKernelFn, typename... TArgs>
    class Kernel
    {
    public:
        Kernel(TKernelFn const& kernelFn, TArgs&&... args)
            : m_kernelFn(kernelFn)
            , m_args(std::forward<TArgs>(args)...)
        {

        }

        using Acc = TAcc;
        using KernelFn = TKernelFn;
        using ArgTupel = std::tuple<remove_restrict_t<std::decay_t<TArgs>>...>;

        KernelFn m_kernelFn;
        ArgTupel m_args;
    };

    template<typename TAcc, typename TKernelFn, typename... TArgs>
    inline auto makeKernel(TKernelFn const& kernelFn, TArgs&&... args)
    {
        return Kernel<TAcc, TKernelFn, TArgs...>(kernelFn, std::forward<TArgs>(args)...);
    };

} // namespace alpaka
