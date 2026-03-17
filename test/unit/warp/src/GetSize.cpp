/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#include <aV2P/test/KernelExecutionFixture.hpp>
#include <aV2P/test/acc/TestAccs.hpp>
#include <aV2P/test/queue/Queue.hpp>
#include <aV2P/warp/Traits.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

template<std::uint32_t TWarpSize>
struct GetSizeTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, std::int32_t expectedWarpSize) const -> void
    {
        // run-time warp size
        std::int32_t size = aV2P::warp::getSize(acc);
        ALPAKA_CHECK(*success, size != 0);
        // compare with the warp size requested when launching the kernel
        ALPAKA_CHECK(*success, size == expectedWarpSize);
        // compare with the compile-time upper limit on the warp size
        constexpr std::int32_t upperLimit = aV2P::warp::getSizeUpperLimit<TAcc>();
        ALPAKA_CHECK(*success, size <= upperLimit);
        // if the warp size is defined at compile-time, compare it with the run-time value
        constexpr std::int32_t constexprSize = aV2P::warp::getSizeCompileTime<TAcc>();
        if constexpr(constexprSize != 0)
        {
            ALPAKA_CHECK(*success, size == constexprSize);
        }
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct aV2P::trait::WarpSize<GetSizeTestKernel<TWarpSize>, TAcc> : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("getSize", "[warp]", aV2P::test::TestAccs)
{
    using Acc = TestType;
    using Dim = aV2P::Dim<Acc>;
    using Idx = aV2P::Idx<Acc>;

    auto const platform = aV2P::Platform<Acc>{};
    auto const dev = aV2P::getDevByIdx(platform, 0);
    auto const warpSizes = aV2P::getWarpSizes(dev);
    REQUIRE(std::any_of(
        begin(warpSizes),
        end(warpSizes),
        [](std::size_t ws)
        {
            aV2P::test::KernelExecutionFixture<Acc> fixture(aV2P::Vec<Dim, Idx>::all(8));
            if(ws == 4)
            {
                return fixture(GetSizeTestKernel<4>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 8)
            {
                return fixture(GetSizeTestKernel<8>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 16)
            {
                return fixture(GetSizeTestKernel<16>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 32)
            {
                return fixture(GetSizeTestKernel<32>{}, static_cast<std::int32_t>(ws));
            }
            else if(ws == 64)
            {
                return fixture(GetSizeTestKernel<64>{}, static_cast<std::int32_t>(ws));
            }
            return fixture(GetSizeTestKernel<0>{}, static_cast<std::int32_t>(ws));
        }));
}
