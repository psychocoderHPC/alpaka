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
    class DevAlpaka3Cpu;
    struct PlatformAlpaka3Cpu;

    namespace trait
    {
        template<typename TPlatform, typename TSfinae>
        struct GetDevByIdx;
    } // namespace trait
    struct PlatformAlpaka3Cpu;

    //! The CPU device handle.
    class DevAlpaka3Cpu
        : public interface::Implements<ConceptCurrentThreadWaitFor, DevAlpaka3Cpu>
        , public interface::Implements<ConceptDev, DevAlpaka3Cpu>
    {
        friend struct trait::GetDevByIdx<PlatformAlpaka3Cpu>;

    protected:
        DevAlpaka3Cpu(size_t devIdx) : m_devImpl{alpaka::onHost::makeDeviceSelector(DeviceSpec{}).makeDevice(devIdx)}
        {
        }

    public:
        using DeviceSpec = alpaka::onHost::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>;
        using DevType = decltype(alpaka::onHost::makeDeviceSelector(DeviceSpec{}).makeDevice(0));

        auto getAlpaka3() const
        {
            return m_devImpl;
        }

        auto operator==(DevAlpaka3Cpu const& other) const -> bool
        {
            return m_devImpl == other.m_devImpl;
        }

        auto operator!=(DevAlpaka3Cpu const& other) const -> bool
        {
            return m_devImpl != other.m_devImpl;
        }

        [[nodiscard]] auto getNativeHandle() const noexcept
        {
            return 0;
        }

        static void setDeviceProperties(alpaka::DevAlpaka3Cpu const& device, alpaka::DeviceProperties& devProperties)
        {
            devProperties.name = device.m_devImpl.getName();
            devProperties.totalGlobalMem = device.m_devImpl.getDeviceProperties().globalMemCapacityBytes;
        }

        friend struct trait::GetName<DevAlpaka3Cpu>;
        friend struct trait::GetMemBytes<DevAlpaka3Cpu>;
        friend struct trait::GetFreeMemBytes<DevAlpaka3Cpu>;
        friend struct trait::GetWarpSizes<DevAlpaka3Cpu>;
        friend struct trait::GetPreferredWarpSize<DevAlpaka3Cpu>;

    private:
        DevType m_devImpl;
    };

    namespace trait
    {

        //! The CPU device name get trait specialization.
        template<>
        struct GetName<DevAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto getName(DevAlpaka3Cpu const& dev) -> std::string
            {
                return dev.m_devImpl.getName();
            }
        };

        //! The CPU device available memory get trait specialization.
        template<>
        struct GetMemBytes<DevAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto getMemBytes(DevAlpaka3Cpu const& dev) -> std::size_t
            {
                return dev.m_devImpl.getDeviceProperties().globalMemCapacityBytes;
            }
        };

        //! The CPU device free memory get trait specialization.
        template<>
        struct GetFreeMemBytes<DevAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevAlpaka3Cpu const& dev) -> std::size_t
            {
                return dev.m_devImpl.getFreeGlobalMemBytes();
            }
        };

        //! The CPU device warp size get trait specialization.
        template<>
        struct GetWarpSizes<DevAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto getWarpSizes(DevAlpaka3Cpu const& dev) -> std::vector<std::size_t>
            {
                return {dev.m_devImpl.getDeviceProperties().warpSize};
            }
        };

        //! The CPU device preferred warp size get trait specialization.
        template<>
        struct GetPreferredWarpSize<DevAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto getPreferredWarpSize(DevAlpaka3Cpu const& dev) -> std::size_t
            {
                return dev.m_devImpl.getDeviceProperties().warpSize;
            }
        };

        //! The CPU device reset trait specialization.
        template<>
        struct Reset<DevAlpaka3Cpu>
        {
            ALPAKA_FN_HOST static auto reset(DevAlpaka3Cpu const& /* dev */) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                // The CPU does nothing on reset.
            }
        };

        //! The CPU device native handle type trait specialization.
        template<>
        struct NativeHandle<DevAlpaka3Cpu>
        {
            [[nodiscard]] static auto getNativeHandle(DevAlpaka3Cpu const& dev)
            {
                return dev.getNativeHandle();
            }
        };
    } // namespace trait

    namespace trait
    {
        //! The CPU device platform type trait specialization.
        template<>
        struct PlatformType<DevAlpaka3Cpu>
        {
            using type = PlatformAlpaka3Cpu;
        };
    } // namespace trait

    namespace trait
    {
        template<>
        struct QueueType<DevAlpaka3Cpu, Blocking>
        {
            using type = QueueAlpaka3CpuBlocking<DevAlpaka3Cpu>;
        };

        template<>
        struct QueueType<DevAlpaka3Cpu, NonBlocking>
        {
            using type = QueueAlpaka3CpuNonBlocking<DevAlpaka3Cpu>;
        };
    } // namespace trait
} // namespace alpaka
