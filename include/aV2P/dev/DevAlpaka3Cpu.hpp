/* Copyright 2024 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber,
 *                Antonio Di Pilato, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/dev/Traits.hpp"
#include "aV2P/dev/common/DeviceProperties.hpp"
#include "aV2P/mem/buf/Traits.hpp"
#include "aV2P/platform/Traits.hpp"
#include "aV2P/queue/Properties.hpp"
#include "aV2P/queue/QueueAlpaka3.hpp"
#include "aV2P/queue/Traits.hpp"
#include "aV2P/traits/Traits.hpp"
#include "aV2P/wait/Traits.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace aV2P
{

    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3;

    namespace trait
    {
        template<typename TPlatform, typename TSfinae>
        struct GetDevByIdx;
    } // namespace trait

    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3;

    template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
    class DevAlpaka3
        : public interface::Implements<ConceptCurrentThreadWaitFor, DevAlpaka3<TDevSpec>>
        , public interface::Implements<ConceptDev, DevAlpaka3<TDevSpec>>
    {
        friend struct trait::GetDevByIdx<PlatformAlpaka3<TDevSpec>>;

    protected:
        DevAlpaka3(size_t devIdx)
            : m_devImpl{alpaka::onHost::makeDeviceSelector(DevSpec{}).makeDevice(static_cast<uint32_t>(devIdx))}
        {
        }

    public:
        using DevSpec = TDevSpec;
        using DevType = decltype(alpaka::onHost::makeDeviceSelector(DevSpec{}).makeDevice(0));

        auto getAlpaka3() const
        {
            return m_devImpl;
        }

        auto operator==(DevAlpaka3 const& other) const -> bool
        {
            return m_devImpl == other.m_devImpl;
        }

        auto operator!=(DevAlpaka3 const& other) const -> bool
        {
            return m_devImpl != other.m_devImpl;
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return getAlpaka3().getNativeHandle();
        }

        friend struct trait::GetName<DevAlpaka3>;
        friend struct trait::GetMemBytes<DevAlpaka3>;
        friend struct trait::GetFreeMemBytes<DevAlpaka3>;
        friend struct trait::GetWarpSizes<DevAlpaka3>;
        friend struct trait::GetPreferredWarpSize<DevAlpaka3>;

    private:
        DevType m_devImpl;
    };

    namespace trait
    {
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct GetName<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getName(DevAlpaka3<TDevSpec> const& dev) -> std::string
            {
                return dev.getAlpaka3().getName();
            }
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct GetMemBytes<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getMemBytes(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3().getDeviceProperties().globalMemCapacityBytes;
            }
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct GetFreeMemBytes<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3().getFreeGlobalMemBytes();
            }
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct GetWarpSizes<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getWarpSizes(DevAlpaka3<TDevSpec> const& dev) -> std::vector<std::size_t>
            {
                return {dev.getAlpaka3().getDeviceProperties().warpSize};
            }
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct GetPreferredWarpSize<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getPreferredWarpSize(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3().getDeviceProperties().warpSize;
            }
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct Reset<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto reset(DevAlpaka3<TDevSpec> const& /* dev */) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                static_assert(sizeof(TDevSpec) && false, "not implemented");
                // The CPU does nothing on reset.
            }
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct NativeHandle<DevAlpaka3<TDevSpec>>
        {
            [[nodiscard]] static auto getNativeHandle(DevAlpaka3<TDevSpec> const& dev)
            {
                return dev.getNativeHandle();
            }
        };
    } // namespace trait

    namespace trait
    {
        //! The CPU device platform type trait specialization.
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct PlatformType<DevAlpaka3<TDevSpec>>
        {
            using type = PlatformAlpaka3<TDevSpec>;
        };
    } // namespace trait

    namespace trait
    {
        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct QueueType<DevAlpaka3<TDevSpec>, Blocking>
        {
            using type = QueueAlpaka3<DevAlpaka3<TDevSpec>, alpaka::queueKind::Blocking>;
        };

        template<alpaka::onHost::concepts::DeviceSpec TDevSpec>
        struct QueueType<DevAlpaka3<TDevSpec>, NonBlocking>
        {
            using type = QueueAlpaka3<DevAlpaka3<TDevSpec>, alpaka::queueKind::NonBlocking>;
        };
    } // namespace trait
} // namespace aV2P
