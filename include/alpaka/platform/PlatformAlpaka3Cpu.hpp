/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Interface.hpp"
#include "alpaka/dev/DevAlpaka3Cpu.hpp"
#include "alpaka/platform/Traits.hpp"

#include <sstream>
#include <vector>

namespace alpaka
{


    //! The CPU device platform.
    struct PlatformAlpaka3Cpu : interface::Implements<ConceptPlatform, PlatformAlpaka3Cpu>
    {
        using DeviceSpec = alpaka::onHost::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>;
    };

    namespace trait
    {
        //! The CPU device device type trait specialization.
        template<>
        struct DevType<PlatformAlpaka3Cpu>
        {
            using type = DevAlpaka3Cpu;
        };

        //! The CPU platform device count get trait specialization.
        template<>
        struct GetDevCount<PlatformAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto getDevCount(PlatformAlpaka3Cpu const&) -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::onHost::makeDeviceSelector(typename PlatformAlpaka3Cpu::DeviceSpec{}).getDeviceCount();
            }
        };

        //! The CPU platform device get trait specialization.
        template<>
        struct GetDevByIdx<PlatformAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto getDevByIdx(PlatformAlpaka3Cpu const&, std::size_t const& devIdx)
                -> DevAlpaka3Cpu
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return DevAlpaka3Cpu{devIdx};
            }
        };
    } // namespace trait
} // namespace alpaka
