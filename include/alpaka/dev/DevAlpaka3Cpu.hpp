/* Copyright 2024 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber,
 *                Antonio Di Pilato, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/Traits.hpp"
#include "alpaka/dev/common/DeviceProperties.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/Properties.hpp"
#include "alpaka/queue/QueueAlpaka3CpuBlocking.hpp"
#include "alpaka/queue/QueueAlpaka3CpuNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/traits/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <alpaka/alpaka3.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace alpaka
{

    template<onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3;

    namespace trait
    {
        template<typename TPlatform, typename TSfinae>
        struct GetDevByIdx;
    } // namespace trait

    template<onHost::concepts::DeviceSpec TDevSpec>
    struct PlatformAlpaka3;

    template<onHost::concepts::DeviceSpec TDevSpec>
    class DevAlpaka3
        : public interface::Implements<ConceptCurrentThreadWaitFor, DevAlpaka3<TDevSpec>>
        , public interface::Implements<ConceptDev, DevAlpaka3<TDevSpec>>
    {
        friend struct trait::GetDevByIdx<PlatformAlpaka3<TDevSpec>>;

    protected:
        DevAlpaka3(size_t devIdx) : m_devImpl{alpaka::onHost::makeDeviceSelector(DevSpec{}).makeDevice(devIdx)}
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
        template<onHost::concepts::DeviceSpec TDevSpec>
        struct GetName<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getName(DevAlpaka3<TDevSpec> const& dev) -> std::string
            {
                return dev.getAlpaka3.getName();
            }
        };

        template<onHost::concepts::DeviceSpec TDevSpec>
        struct GetMemBytes<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getMemBytes(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3.getDeviceProperties().globalMemCapacityBytes;
            }
        };

        template<onHost::concepts::DeviceSpec TDevSpec>
        struct GetFreeMemBytes<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3.getFreeGlobalMemBytes();
            }
        };

        template<onHost::concepts::DeviceSpec TDevSpec>
        struct GetWarpSizes<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getWarpSizes(DevAlpaka3<TDevSpec> const& dev) -> std::vector<std::size_t>
            {
                return {dev.getAlpaka3.getDeviceProperties().warpSize};
            }
        };

        template<onHost::concepts::DeviceSpec TDevSpec>
        struct GetPreferredWarpSize<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto getPreferredWarpSize(DevAlpaka3<TDevSpec> const& dev) -> std::size_t
            {
                return dev.getAlpaka3.getDeviceProperties().warpSize;
            }
        };

        template<onHost::concepts::DeviceSpec TDevSpec>
        struct Reset<DevAlpaka3<TDevSpec>>
        {
            ALPAKA_FN_HOST static auto reset(DevAlpaka3<TDevSpec> const& /* dev */) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                static_assert(sizeof(TDevSpec) && false, "not implemented");
                // The CPU does nothing on reset.
            }
        };

        template<onHost::concepts::DeviceSpec TDevSpec>
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
        template<onHost::concepts::DeviceSpec TDevSpec>
        struct PlatformType<DevAlpaka3<TDevSpec>>
        {
            using type = PlatformAlpaka3<TDevSpec>;
        };
    } // namespace trait

    namespace trait
    {
        template<onHost::concepts::DeviceSpec TDevSpec>
        struct QueueType<DevAlpaka3<TDevSpec>, Blocking>
        {
            using type = QueueAlpaka3CpuBlocking<DevAlpaka3<TDevSpec>>;
        };

        template<onHost::concepts::DeviceSpec TDevSpec>
        struct QueueType<DevAlpaka3<TDevSpec>, NonBlocking>
        {
            using type = QueueAlpaka3CpuNonBlocking<DevAlpaka3<TDevSpec>>;
        };
    } // namespace trait
} // namespace alpaka
