/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/core/DemangleTypeNames.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <tuple>
#include <typeinfo>

//! A vector addition kernel.
class VectorAddKernel
{
public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TElem The matrix element type.
    //! \param acc The accelerator to be executed on.
    //! \param A The first source vector.
    //! \param B The second source vector.
    //! \param C The destination vector.
    //! \param numElements The number of elements.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TElem, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const A,
        TElem const* const B,
        TElem* const C,
        TIdx const& numElements) const -> void
    {
        static_assert(alpaka::Dim<TAcc>::value == 1, "The VectorAddKernel expects 1-dimensional indices!");

        // The uniformElements range for loop takes care automatically of the blocks, threads and elements in the
        // kernel launch grid.
        for(auto i : alpaka::uniformElements(acc, numElements))
        {
            C[i] = A[i] + B[i];
        }
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&, size_t numElements) -> int
{
    // Define the index domain
    // Set the number of dimensions as an integral constant. Set to 1 for 1D.
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    // Define the buffer element type
    using Data = std::uint32_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;

    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Element type: " << std::string(alpaka::core::demangled<Data>) << std::endl;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    //
    // choose between Blocking and NonBlocking
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);

    // Create a queue on the device
    QueueAcc queue(devAcc);

    // Define the work division
    Idx const elementsPerThread(8u);
    alpaka::Vec<Dim, Idx> const extent(numElements);


    // Get the host device for allocating memory on the host.
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostB(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostC(alpaka::allocBuf<Data, Idx>(devHost, extent));

    // C++14 random generator for uniformly distributed numbers in {1,..,42}
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<Data> dist(1, 42);

    for(Idx i(0); i < numElements; ++i)
    {
        bufHostA[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostC[i] = 0;
    }

    // Allocate 3 buffers on the accelerator
    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccC(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    // Copy Host -> Acc
    alpaka::memcpy(queue, bufAccA, bufHostA);
    alpaka::memcpy(queue, bufAccB, bufHostB);
    alpaka::memcpy(queue, bufAccC, bufHostC);

    // Instantiate the kernel function object
    VectorAddKernel kernel;

    alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        kernel,
        alpaka::getPtrNative(bufAccA),
        alpaka::getPtrNative(bufAccB),
        alpaka::getPtrNative(bufAccC),
        numElements);

    // Create the kernel execution task.
    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        kernel,
        std::data(bufAccA),
        std::data(bufAccB),
        std::data(bufAccC),
        numElements);

    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    int falseResults = 0;

    constexpr uint32_t numRounds = 10;
    double elapsedTime = 0;
    for(uint32_t rounds = 0; rounds < numRounds; ++rounds)
    {
        // Enqueue the kernel execution task
        {
            // wait in case we are using an asynchronous queue to time actual kernel runtime
            alpaka::wait(queue);
            auto const beginT = std::chrono::high_resolution_clock::now();
            alpaka::enqueue(queue, taskKernel);
            // wait in case we are using an asynchronous queue to time actual kernel runtime
            alpaka::wait(queue);
            auto const endT = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> foo = (endT - beginT);
            elapsedTime += foo.count();
        }

        if(rounds == 0)
        {
            // Copy back the result
            {
                auto beginT = std::chrono::high_resolution_clock::now();
                alpaka::memcpy(queue, bufHostC, bufAccC);
                alpaka::wait(queue);
                auto const endT = std::chrono::high_resolution_clock::now();
                std::cout << "Time for HtoD copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                          << std::endl;
            }

            for(Idx i(0u); i < numElements; ++i)
            {
                Data const& val(bufHostC[i]);
                Data const correctResult(bufHostA[i] + bufHostB[i]);
                if(val != correctResult)
                {
                    if(falseResults < MAX_PRINT_FALSE_RESULTS)
                        std::cerr << "C[" << i << "] == " << val << " != " << correctResult << std::endl;
                    ++falseResults;
                }
            }
        }
    }
    std::cout << "runtime " << elapsedTime << " seconds." << std::endl;
    std::cout << "benchmark:" << numElements << "," << elapsedTime << std::endl;
    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [-n  numElements] [-h]" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    size_t numElements = 123456;

    int opt;
    while((opt = getopt(argc, argv, "hn:")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    constexpr uint32_t numAccs = std::tuple_size_v<decltype(alpaka::EnabledAccTags{})>;
    if constexpr(numAccs == 1)
        return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag, numElements); });
    else
        return example(std::get < numAccs == 1 ? 0 : 1 > (alpaka::EnabledAccTags{}), numElements);
}
