/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>

#include <vector>


//-----------------------------------------------------------------------------
//! test that all threads perform the a kernel collective
struct TestCollective
{

void operator()()
{
    // Define the index domain
    using Dim = alpaka::dim::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = alpaka::dev::Dev<Acc>;

    using Queue = alpaka::queue::QueueCpuOmp2Collective;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    auto dev = alpaka::pltf::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(results.size());

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);

    #pragma omp parallel num_threads(static_cast<int>(results.size()))
    {
        auto kernel =
        [&] ALPAKA_FN_ACC (
            Acc const & acc) noexcept
        -> void
        {
            size_t threadId = alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
            // avoid that one thread is doing all the work
            std::this_thread::sleep_for(std::chrono::milliseconds(200u * threadId));
            results[threadId] = threadId;
        };

        alpaka::kernel::exec<Acc>(
               queue,
               workDiv,
               kernel);

        alpaka::wait::wait(queue);
    }

    for(size_t i = 0; i < results.size(); ++i)
    {
        REQUIRE(static_cast<int>(i) == results.at(i));
    }
}
};

//-----------------------------------------------------------------------------
//! test that only one thread performs the copy operation
struct TestCollectiveMemcpy
{

void operator()()
{
    // Define the index domain
    using Dim = alpaka::dim::DimInt<1>;
    using Idx = size_t;

    // Define the accelerator
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Idx>;
    using Dev = alpaka::dev::Dev<Acc>;

    using Queue = alpaka::queue::QueueCpuOmp2Collective;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    auto dev = alpaka::pltf::getDevByIdx<Pltf>(0u);
    Queue queue(dev);

    std::vector<int> results(4, -1);

    // Define the work division
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<Idx>(1)));
    Vec const blocksPerGrid(results.size());

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    WorkDiv const workDiv(
        blocksPerGrid,
        threadsPerBlock,
        elementsPerThread);

    #pragma omp parallel num_threads(static_cast<int>(results.size()))
    {
        int threadId = omp_get_thread_num();

        using View = alpaka::mem::view::ViewPlainPtr<Dev, int, Dim, Idx>;

        View dst(
            results.data() + threadId,
            dev,
            Vec(1lu),
            Vec(sizeof(int)));

        View src(
            &threadId,
            dev,
            Vec(1lu),
            Vec(sizeof(int)));

        // avoid that the first thread is executing the copy (can not be guaranteed)
        size_t sleep_ms = (results.size() - static_cast<uint32_t>(threadId)) * 100u;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        alpaka::mem::view::copy(queue, dst, src, Vec(1lu));

        alpaka::wait::wait(queue);
    }

    uint32_t numFlippedValues = 0u;
    uint32_t numNonIntitialValues = 0u;
    for(size_t i = 0; i < results.size(); ++i)
    {
        if(static_cast<int>(i) == results.at(i))
            numFlippedValues++;
        if(results.at(i) != -1)
            numNonIntitialValues++;
    }
    // only one thread is allowed to flip the value
    REQUIRE(numFlippedValues == 1u);
    // only one value is allowed to differ from the initial value
    REQUIRE(numNonIntitialValues == 1u);
}
};

TEST_CASE( "queueCollective", "[queue]")
{
    TestCollective{}();
}

TEST_CASE( "TestCollectiveMemcpy", "[queue]")
{
    TestCollectiveMemcpy{}();
}

#endif
