/* Copyright 2019 Rene Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>

#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <iostream>
#include <typeinfo>
#include <vector>
#include <functional>

struct MyKernel
{
   template<typename Acc>
   ALPAKA_FN_ACC void operator()(Acc const & acc, const double* sourceData, size_t* errorCounter) const
   {
     constexpr size_t size = 900 * 5 * 5;
     int i = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
     // note (same as for CUDA): here we are supposed to check that i is in the array range
     // but this is not what is causing the issue
     if(i < size && sourceData[i] != 1.0)
     {
        alpaka::atomic::atomicOp<
             alpaka::atomic::op::Add>(
                 acc,
                 errorCounter,
                 std::size_t(1u));
     }
   }
};

struct EmptyKernel
{
   template<typename Acc>
   ALPAKA_FN_ACC void operator()(Acc const & acc, int threadElementExtent) const
   {
     assert(threadElementExtent == 1);
   }
};

struct TestTemplate
{
    template< typename TAcc >
    void operator()()
    {
        const size_t size = 900 * 5 * 5;

        using ComputeAccelerator = TAcc;
        using ComputeDevice = alpaka::dev::Dev<ComputeAccelerator>;
        using ComputeStream = alpaka::test::queue::DefaultQueue<alpaka::dev::Dev<TAcc>>;

        using Dim = alpaka::dim::DimInt<1u>;
        using Idx = std::size_t;

        ComputeDevice computeDevice(alpaka::pltf::getDevByIdx<alpaka::pltf::Pltf<ComputeDevice> >(0));
        ComputeStream computeStream (computeDevice);

        using HostDevice = alpaka::dev::DevCpu;
        using PltfHost = alpaka::pltf::Pltf<HostDevice>;
        HostDevice const hostDevice(alpaka::pltf::getDevByIdx<PltfHost>(0u));

        using HostViewType = alpaka::mem::view::ViewPlainPtr<
            HostDevice, alpaka::elem::Elem<double>, Dim, Idx >;

        using HostViewErrorType = alpaka::mem::view::ViewPlainPtr<
            HostDevice, alpaka::elem::Elem<size_t>, Dim, Idx >;

        using V = alpaka::vec::Vec<alpaka::dim::DimInt<1>, std::size_t>;

        auto devErrorCounter = alpaka::mem::buf::alloc<size_t, size_t>(computeDevice, V(std::size_t(1u)));
        auto hostErrorCounter = alpaka::mem::buf::alloc<size_t, size_t>(hostDevice, V(std::size_t(1u)));
        *alpaka::mem::view::getPtrNative(hostErrorCounter) = 0u;

        alpaka::mem::view::copy(computeStream, devErrorCounter, hostErrorCounter, V(std::size_t(1u)));

        using WorkDivision = alpaka::workdiv::WorkDivMembers<alpaka::dim::DimInt<1>, std::size_t>;
        WorkDivision wd(V(std::size_t(((size - 1) / 64) + 1)), V(std::size_t(64)), V(std::size_t(1)));

        alpaka::vec::Vec<alpaka::dim::DimInt<1>, size_t> bufferSize (size);

        auto sourceMem = alpaka::mem::buf::alloc<double, size_t>(computeDevice, size);

        alpaka::queue::enqueue(computeStream, alpaka::kernel::createTaskKernel<ComputeAccelerator>(wd, EmptyKernel(), 1));

        std::vector<double> sourceMemHost(size, 1.0);
        HostViewType hostBufferView(sourceMemHost.data(), hostDevice, bufferSize);
        alpaka::mem::view::copy(computeStream, sourceMem, hostBufferView, bufferSize);
        alpaka::wait::wait(computeStream);

        alpaka::queue::enqueue(computeStream,
          alpaka::kernel::createTaskKernel<ComputeAccelerator>(wd, MyKernel(), alpaka::mem::view::getPtrNative(sourceMem), alpaka::mem::view::getPtrNative(devErrorCounter)));

        alpaka::mem::view::copy(computeStream, hostErrorCounter, devErrorCounter, V(std::size_t(1u)));
        alpaka::wait::wait(computeStream);

        alpaka::wait::wait(computeDevice);
        auto errorCount = *alpaka::mem::view::getPtrNative(hostErrorCounter);

        REQUIRE(errorCount == 0);
    }
};



TEST_CASE( "matMul", "[matMul]")
{
    using TestAccs = alpaka::test::acc::EnabledAccs<
        alpaka::dim::DimInt<1u>,
        std::size_t>;

    alpaka::meta::forEachType< TestAccs >( TestTemplate() );
}
