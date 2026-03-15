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
    template<onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3 : interface::Implements<ConceptPlatform, PlatformAlpaka3<TDevSpec>>
    {
        using DevSpec = alpaka::onHost::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>;
    };

    using PlatformAlpaka3Cpu = PlatformAlpaka3<onHost::DeviceSpec<api::Host, deviceKind::Cpu>>;

    namespace trait
    {
        //! The CPU device device type trait specialization.
        template<onHost::concepts::DeviceSpec TDevSpec>
        struct DevType<PlatformAlpaka3<TDevSpec>>
        {
            using type = DevAlpaka3<TDevSpec>;
        };

        //! The CPU platform device count get trait specialization.
        template<onHost::concepts::DeviceSpec TDevSpec>
        struct GetDevCount<PlatformAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getDevCount(PlatformAlpaka3Cpu const&) -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::onHost::makeDeviceSelector(TDevSpec{}).getDeviceCount();
            }
        };

        //! The CPU platform device get trait specialization.
        template<onHost::concepts::DeviceSpec TDevSpec>
        struct GetDevByIdx<PlatformAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getDevByIdx(PlatformAlpaka3Cpu const&, std::size_t const& devIdx)
                -> DevAlpaka3<TDevSpec>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return DevAlpaka3<TDevSpec>{devIdx};
            }
        };
    } // namespace trait
} // namespace alpaka
