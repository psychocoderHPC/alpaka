/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>                        // alpaka::exec::create

#include <iostream>                                 // std::cout
#include <string>
#include <typeinfo>                                 // typeid
#include <chrono>                       // std::chrono::high_resolution_clock


using Val = float;
using Size = std::size_t;
using Dim = alpaka::dim::DimInt<1u>;

using PltfHost = alpaka::pltf::PltfCpu;

/** dummy kernel to initialize the device
 */
class Dummy
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & ) const
    -> void
    {
    }
};

//#############################################################################
//! A vector addition kernel.
//! \tparam TAcc The accelerator environment to be executed on.
//#############################################################################
class VectorAddKernel
{
public:
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //!
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem,
        typename TSize>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        TElem const * const A,
        TElem const * const B,
        TElem * const C,
        TSize const & numElements) const
    -> void
    {
        static_assert(
            alpaka::dim::Dim<TAcc>::value == 1,
            "The VectorAddKernel expects 1-dimensional indices!");

        auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < numElements)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            auto const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            auto const threadLastElemIdxClipped((threadLastElemIdx < numElements) ? threadLastElemIdx : numElements);

            for(TSize i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                C[i] = A[i] + B[i];
            }
        }
    }
};

template<typename TAcc, typename TStream>
class AccHandler
{

    using Acc = TAcc;
    using Stream = TStream;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<Acc>;

    using SubView = alpaka::mem::view::ViewSubView<
        alpaka::dev::Dev<alpaka::acc::AccCpuSerial<Dim, Size> >,
        Val,
        Dim,
        Size
    >;

    DevAcc const devAcc;
    Stream stream;
    alpaka::Vec<Dim, Size> const m_extent;

    using BufferType = decltype(alpaka::mem::buf::alloc<Val, Size>(devAcc, m_extent));

    BufferType memBufA;
    BufferType memBufB;
    BufferType memBufC;

public:

    AccHandler(Size const extent, int deviceNumber = 0) :
        devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(deviceNumber)),
        stream(devAcc),
        m_extent(extent),
        memBufA(alpaka::mem::buf::alloc<Val, Size>(devAcc, m_extent)),
        memBufB(alpaka::mem::buf::alloc<Val, Size>(devAcc, m_extent)),
        memBufC(alpaka::mem::buf::alloc<Val, Size>(devAcc, m_extent))
    {
        /* execute a dummy kernel to avoid initialize overhead while
         * time measurement
         */
        alpaka::stream::enqueue(
            stream,
            alpaka::exec::create<Acc>(
                alpaka::workdiv::WorkDivMembers<
                    Dim,
                    Size
                >(
                    alpaka::Vec<Dim, Size>(static_cast<Size>(1)),
                    alpaka::Vec<Dim, Size>(static_cast<Size>(1)),
                    alpaka::Vec<Dim, Size>(static_cast<Size>(1))
                ),
                Dummy()
            )
        );

        // Allocate the buffers on the accelerator.
        //memBufA = alpaka::mem::buf::alloc<Val, Size>(devAcc, m_extent);
        //memBufB = alpaka::mem::buf::alloc<Val, Size>(devAcc, m_extent);
        //memBufC = alpaka::mem::buf::alloc<Val, Size>(devAcc, m_extent);
    }

    template<typename TBuffer>
    void copyToDevice(TBuffer& memBufAHost, TBuffer& memBufBHost, Size const& globalOffset = Size(0) )
    {
        alpaka::Vec<Dim, Size> const offset(globalOffset);

        SubView viewMemAHost(memBufAHost, m_extent, offset);
        SubView viewMemBHost(memBufBHost, m_extent, offset);

        // Copy Host -> Acc.
        alpaka::mem::view::copy(stream, memBufA, viewMemAHost, m_extent);
        alpaka::mem::view::copy(stream, memBufB, viewMemBHost, m_extent);
    }

    template<typename TBuffer>
    void copyToHost(TBuffer& destHost, Size const& globalOffset = Size(0) )
    {
        alpaka::Vec<Dim, Size> const offset(globalOffset);

        SubView viewDestHost(destHost, m_extent, offset);

        // Copy back the result.
        alpaka::mem::view::copy(stream, viewDestHost, memBufC, m_extent);
    }


    void execute( )
    {
        // return if accelerator has no work
        if(m_extent[0]==Size(0))
            return;
        // Create the kernel function object.
        VectorAddKernel kernel;

        // Let alpaka calculate good block and grid sizes given our full problem extent.
        alpaka::workdiv::WorkDivMembers<Dim, Size> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                m_extent,
                static_cast<Size>(4u),
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

        std::cout
            << "VectorAddKernelTester("
            << " numElements:" << m_extent[0]
            << ", accelerator: " << alpaka::acc::getAccName<TAcc>()
            << ", kernel: " << typeid(kernel).name()
            << ", workDiv: " << workDiv
            << ")" << std::endl;

            // Create the executor task.
        auto const exec(alpaka::exec::create<Acc>(
            workDiv,
            kernel,
            alpaka::mem::view::getPtrNative(memBufA),
            alpaka::mem::view::getPtrNative(memBufB),
            alpaka::mem::view::getPtrNative(memBufC),
            m_extent[0]));

         // vector add the kernel execution.
        alpaka::stream::enqueue(stream, exec);
    }

    void wait()
    {
        alpaka::wait::wait(stream);
    }
};


//-----------------------------------------------------------------------------
//! Program entry point.
//-----------------------------------------------------------------------------
auto main(int argc,  char *argv[])
-> int
{
    /** accelerator types
     * - AccGpuCudaRt
     * - AccCpuThreads
     * - AccCpuOmp2Threads
     * - AccCpuOmp2Blocks
     * - AccCpuSerial
     */
    using Acc1 = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    using Acc2 = alpaka::acc::AccGpuCudaRt<Dim, Size>;


    /** possible stream versions
     *
     * CPU:
     *   - StreamCpuAsync
     *   - StreamCpuSync
     * GPU:
     *   - StreamCudaRtAsync
     *   - StreamCudaRtSync
     */
    using StreamAcc1 = alpaka::stream::StreamCpuAsync;
    using StreamAcc2 = alpaka::stream::StreamCudaRtAsync;


    Size numElements(123456);
    double percentAcc1 = 10.;

    // check for user input
    if( argc >= 2 )
    {
        numElements = std::stoi(argv[1]);
    }
    std::cout<<"num elements: "<<numElements<<std::endl;

    // check for user input
    if( argc == 3 )
    {
        percentAcc1 = std::stod(argv[2]);
    }

    // Get the host device.
    auto const devHost(
        alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // The data extent.
    alpaka::Vec<Dim, Size> const extent(
        numElements);

    // Allocate host memory buffers.
    auto memBufHostA(alpaka::mem::buf::alloc<Val, Size>(devHost, extent));
    auto memBufHostB(alpaka::mem::buf::alloc<Val, Size>(devHost, extent));
    auto memBufHostC(alpaka::mem::buf::alloc<Val, Size>(devHost, extent));

    // Initialize the host input vectors
    for (Size i(0); i < numElements; ++i)
    {
        alpaka::mem::view::getPtrNative(memBufHostA)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
        alpaka::mem::view::getPtrNative(memBufHostB)[i] = static_cast<Val>(rand()) / static_cast<Val>(RAND_MAX);
    }


    Size numElemAcc1 = ceil((percentAcc1 * double(numElements))/100.);

    std::cout<<"num elements: acc1="<<numElemAcc1<<" acc2="<<(numElements - numElemAcc1)<<std::endl;

    AccHandler<Acc1,StreamAcc1> acc1(numElemAcc1);
    AccHandler<Acc2,StreamAcc2> acc2(numElements - numElemAcc1);

    // copy data from host to device
    acc1.copyToDevice(memBufHostA, memBufHostB);
    acc2.copyToDevice(memBufHostA, memBufHostB, numElemAcc1);

    /* wait for the stream */
    acc1.wait(); acc2.wait();

    // Take the time prior to the execution.
    auto const execStr(std::chrono::high_resolution_clock::now());

    // execute vector addition
    acc1.execute();
    acc2.execute();

    acc1.wait(); acc2.wait();
    // Take the time after to the execution.
    auto const execEnd(std::chrono::high_resolution_clock::now());

    // copy results from device to host
    acc1.copyToHost(memBufHostC);
    acc2.copyToHost(memBufHostC, numElemAcc1);

    acc1.wait(); acc2.wait();

    // the duration.
    auto const execDur(execEnd - execStr);
    auto durExecution = std::chrono::duration_cast<std::chrono::microseconds>(execDur).count();
    std::cout<<"execution time "<<durExecution<<" us"<<std::endl;

    bool resultCorrect(true);
    auto const pHostData(alpaka::mem::view::getPtrNative(memBufHostC));
    for(Size i(0u);
        i < numElements;
        ++i)
    {
        auto const & val(pHostData[i]);
        auto const correctResult(alpaka::mem::view::getPtrNative(memBufHostA)[i]+alpaka::mem::view::getPtrNative(memBufHostB)[i]);
        if(val != correctResult)
        {
            std::cout << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
            resultCorrect = false;
        }
    }

    if(resultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
    }

    return EXIT_SUCCESS;
}
