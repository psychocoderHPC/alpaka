/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/exec/Once.hpp"

#include "alpaka/atomic/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/meta/ForEachType.hpp"
#include "alpaka/test/KernelExecutionFixture.hpp"
#include "alpaka/test/acc/TestAccs.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

class KernelOncePerGrid
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* status, int32_t* value) const -> void
    {
        // Only one thread in the whole grid should increment the conter.
        if(aV2P::oncePerGrid(acc))
        {
            ALPAKA_CHECK(*status, *value == 0);

            aV2P::atomicAdd(acc, value, 1, aV2P::hierarchy::Grids{});

            ALPAKA_CHECK(*status, *value == 1);
        }
    }
};

// MSVC does not seem to recognize as "true" a value set to "true" in device code,
// so force all object representations different from zero to evaluate as "true".
inline void fixBooleanValue(bool& value)
{
    value = reinterpret_cast<char const&>(value) == 0x00 ? false : true;
}

TEMPLATE_LIST_TEST_CASE("oncePerGrid", "[exec]", aV2P::test::TestAccs)
{
    using Host = aV2P::DevCpu;
    Host host = aV2P::getDevByIdx(aV2P::PlatformCpu{}, 0);

    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Platform = aV2P::Platform<Acc>;
    using Device = aV2P::Dev<Platform>;
    using Queue = aV2P::Queue<Device, aV2P::NonBlocking>;

    Platform platform;
    Device device = aV2P::getDevByIdx(platform, 0);
    Queue queue{device};

    using Scalar = aV2P::Vec<aV2P::DimInt<0u>, Idx>;
    auto value = aV2P::allocMappedBuf<int32_t, Idx>(host, platform, Scalar{});
    *value = 0;

    auto status = aV2P::allocMappedBuf<bool, Idx>(host, platform, Scalar{});
    *status = true;

    auto const extent = aV2P::Vec<Dim, Idx>::all(32);
    auto const elems = aV2P::Vec<Dim, Idx>::all(4);

    KernelOncePerGrid kernel;
    aV2P::KernelCfg<Acc> const config = {extent, elems, false};
    auto const workDiv = aV2P::getValidWorkDiv(config, device, kernel, std::data(status), std::data(value));

    aV2P::exec<Acc>(queue, workDiv, kernel, std::data(status), std::data(value));
    aV2P::wait(queue);

    fixBooleanValue(*status);
    REQUIRE(*status == true);
    REQUIRE(*value == 1);
}

class KernelOncePerBlock
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* status, int32_t* value) const -> void
    {
        const int32_t blocks = static_cast<int32_t>(aV2P::getWorkDiv<aV2P::Grid, aV2P::Blocks>(acc).prod());

        // Only one thread in each block should increment the conter.
        if(aV2P::oncePerBlock(acc))
        {
            // FIXME: implement aV2P::atomicLoad and use it here
            int before = aV2P::atomicAdd(acc, value, 0, aV2P::hierarchy::Grids{});
            ALPAKA_CHECK(*status, before >= 0);
            ALPAKA_CHECK(*status, before < blocks);

            aV2P::atomicAdd(acc, value, 1, aV2P::hierarchy::Grids{});

            // FIXME: implement aV2P::atomicLoad and use it here
            int after = aV2P::atomicAdd(acc, value, 0, aV2P::hierarchy::Grids{});
            ALPAKA_CHECK(*status, after > 0);
            ALPAKA_CHECK(*status, after <= blocks);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("oncePerBlock", "[exec]", aV2P::test::TestAccs)
{
    using Host = aV2P::DevCpu;
    Host host = aV2P::getDevByIdx(aV2P::PlatformCpu{}, 0);

    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;
    using Platform = aV2P::Platform<Acc>;
    using Device = aV2P::Dev<Platform>;
    using Queue = aV2P::Queue<Device, aV2P::NonBlocking>;

    Platform platform;
    Device device = aV2P::getDevByIdx(platform, 0);
    Queue queue{device};

    using Scalar = aV2P::Vec<aV2P::DimInt<0u>, Idx>;
    auto value = aV2P::allocMappedBuf<int32_t, Idx>(host, platform, Scalar{});
    aV2P::memset(queue, value, 0x00);

    auto status = aV2P::allocMappedBuf<bool, Idx>(host, platform, Scalar{});
    aV2P::memset(queue, status, 0xff);

    auto const extent = aV2P::Vec<Dim, Idx>::all(32);
    auto const elems = aV2P::Vec<Dim, Idx>::all(4);

    KernelOncePerBlock kernel;
    aV2P::KernelCfg<Acc> const config = {extent, elems, false};
    auto const workDiv = aV2P::getValidWorkDiv(config, device, kernel, std::data(status), std::data(value));
    const int32_t blocks = static_cast<int32_t>(aV2P::getWorkDiv<aV2P::Grid, aV2P::Blocks>(workDiv).prod());

    aV2P::exec<Acc>(queue, workDiv, kernel, std::data(status), std::data(value));
    aV2P::wait(queue);

    fixBooleanValue(*status);
    REQUIRE(*status == true);
    REQUIRE(*value == blocks);
}
