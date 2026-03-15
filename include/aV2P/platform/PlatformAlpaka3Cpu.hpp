/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Interface.hpp"
#include "aV2P/dev/DevAlpaka3Cpu.hpp"
#include "aV2P/platform/Traits.hpp"

#include <sstream>
#include <vector>

namespace aV2P
{


    //! The CPU device platform.
    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3 : interface::Implements<ConceptPlatform, PlatformAlpaka3<TDevSpec>>
    {
        using DevSpec = alpaka::onHost::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>;
    };

    using PlatformAlpaka3Cpu = PlatformAlpaka3<alpaka::onHost::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>>;

    namespace trait
    {
        //! The CPU device device type trait specialization.
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct DevType<PlatformAlpaka3<TDevSpec>>
        {
            using type = DevAlpaka3<TDevSpec>;
        };

        //! The CPU platform device count get trait specialization.
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct GetDevCount<PlatformAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getDevCount(PlatformAlpaka3Cpu const&) -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::onHost::makeDeviceSelector(TDevSpec{}).getDeviceCount();
            }
        };

        //! The CPU platform device get trait specialization.
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
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
} // namespace aV2P
