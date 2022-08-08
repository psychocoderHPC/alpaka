/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "AtomicFunctors.hpp"

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/math/FloatEqualExact.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <climits>
#include <type_traits>


using namespace alpaka::test::unit::atomic;

template<typename T1, typename T2>
ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto equals(T1 a, T2 b) -> bool
{
    return a == b;
}

ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto equals(float a, float b) -> bool
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto equals(double a, double b) -> bool
{
    return alpaka::math::floatEqualExactNoWarning(a, b);
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename THierarchy, typename TOp, typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicCall(TAcc const& acc, bool* success, T operandOrig, T value) -> void
{
    auto op = typename TOp::Op{};

    auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);

    // check if the function `alpaka::atomicOp<*>` is callable
    {
        operand = operandOrig;
        T reference = operand;
        op(&reference, value);

        T const ret = alpaka::atomicOp<typename TOp::Op>(acc, &operand, value, THierarchy{});
        // check that always the old value is returned
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        // check that result in memory is correct
        ALPAKA_CHECK(*success, equals(operand, reference));
    }

    // check if the function `alpaka::atomic*()` is callable
    {
        operand = operandOrig;
        T reference = operand;
        op(&reference, value);

        T const ret = TOp::atomic(acc, &operand, value, THierarchy{});
        // check that always the old value is returned
        ALPAKA_CHECK(*success, equals(operandOrig, ret));
        // check that result in memory is correct
        ALPAKA_CHECK(*success, equals(operand, reference));
    }
}

template<typename TOp>
class TestAtomicOp
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename THierarchy, typename TAcc, typename T>
    static ALPAKA_FN_ACC auto test(TAcc const& acc, bool* success, T operandOrig) -> void
    {
        {
            // left operand is half of the right
            T const value = static_cast<T>(operandOrig / static_cast<T>(2));
            testAtomicCall<THierarchy, TOp>(acc, success, operandOrig, value);
        }

        ::alpaka::syncBlockThreads(acc);
        {
            // left operand is twice as large as the right
            T const value = static_cast<T>(operandOrig * static_cast<T>(2));
            testAtomicCall<THierarchy, TOp>(acc, success, operandOrig, value);
        }

        ::alpaka::syncBlockThreads(acc);
        {
            // left operand is larger by one
            T const value = static_cast<T>(operandOrig + static_cast<T>(1));
            testAtomicCall<THierarchy, TOp>(acc, success, operandOrig, value);
        }

        ::alpaka::syncBlockThreads(acc);
        {
            // left operand is smaller by one
            T const value = static_cast<T>(operandOrig - static_cast<T>(1));
            testAtomicCall<THierarchy, TOp>(acc, success, operandOrig, value);
        }
        ::alpaka::syncBlockThreads(acc);
        {
            // both operands are equal
            T const value = operandOrig;
            testAtomicCall<THierarchy, TOp>(acc, success, operandOrig, value);
        }

    }
};

template<>
class TestAtomicOp<Cas>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename THierarchy, typename TAcc, typename T>
    static ALPAKA_FN_ACC auto test(TAcc const& acc, bool* success, T operandOrig) -> void
    {
        T const value = static_cast<T>(4);

        auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);

        // with match
        {
            T const compare = operandOrig;
            T const reference = value;
            {
                operand = operandOrig;
                T const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value, THierarchy{});
                ALPAKA_CHECK(*success, equals(operandOrig, ret));
                ALPAKA_CHECK(*success, equals(operand, reference));
            }
            {
                operand = operandOrig;
                T const ret = alpaka::atomicCas(acc, &operand, compare, value, THierarchy{});
                ALPAKA_CHECK(*success, equals(operandOrig, ret));
                ALPAKA_CHECK(*success, equals(operand, reference));
            }
        }

        // without match
        {
            T const compare = static_cast<T>(operandOrig + static_cast<T>(1));
            T const reference = operandOrig;
            {
                operand = operandOrig;
                T const ret = alpaka::atomicOp<alpaka::AtomicCas>(acc, &operand, compare, value, THierarchy{});
                ALPAKA_CHECK(*success, equals(operandOrig, ret));
                ALPAKA_CHECK(*success, equals(operand, reference));
            }
            {
                operand = operandOrig;
                T const ret = alpaka::atomicCas(acc, &operand, compare, value, THierarchy{});
                ALPAKA_CHECK(*success, equals(operandOrig, ret));
                ALPAKA_CHECK(*success, equals(operand, reference));
            }
        }
    }
};

template<typename TOp, typename TAcc, typename T, typename Sfinae = void>
class AtomicTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T operandOrig) const -> void
    {
        TestAtomicOp<TOp>::template test<alpaka::hierarchy::Threads>(acc, success, operandOrig);
        TestAtomicOp<TOp>::template test<alpaka::hierarchy::Blocks>(acc, success, operandOrig);
        TestAtomicOp<TOp>::template test<alpaka::hierarchy::Grids>(acc, success, operandOrig);
    }
};


template<typename TOp, typename TAcc, typename T>
class AtomicTestKernel<
    TOp,
    TAcc,
    T,
    std::enable_if_t<std::is_floating_point_v<T> && !alpaka::meta::Contains<std::tuple<Inc, Dec, Or, And, Cas>, T>::value>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const&, bool* success, T) const -> void
    {
        // Not supported for floating point types
        ALPAKA_CHECK(*success, true);
    }
};


#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)

template<typename TOp, typename TApi, typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
    TOp,
    alpaka::AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
    T,
    std::enable_if_t<sizeof(T) != 4u && sizeof(T) != 8u>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(
        alpaka::AccGpuUniformCudaHipRt<TApi, TDim, TIdx> const& /* acc */,
        bool* success,
        T /* operandOrig */) const -> void
    {
        // Only 32/64bit atomics are supported
        ALPAKA_CHECK(*success, true);
    }
};

#endif

#if defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED)

template<typename TOp, typename TDim, typename TIdx, typename T>
class AtomicTestKernel<TOp, alpaka::AccOacc<TDim, TIdx>, T, std::enable_if_t<sizeof(T) != 4u && sizeof(T) != 8u>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(alpaka::AccOacc<TDim, TIdx> const& /* acc */, bool* success, T /* operandOrig */)
        const -> void
    {
        // Only 32/64bit atomics are supported
        ALPAKA_CHECK(*success, true);
    }
};

#endif

template<typename TAcc, typename T>
struct TestAtomicOperations
{
    static auto testAtomicOperations() -> void
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

        T value = static_cast<T>(32);

        AtomicTestKernel<Add, TAcc, T> kernelAtomicAdd;
        REQUIRE(fixture(kernelAtomicAdd, value));
#if 0
        AtomicTestKernel<Sub, TAcc, T> kernelAtomicSub;
        REQUIRE(fixture(kernelAtomicSub, value));

        AtomicTestKernel<Exch, TAcc, T> kernelAtomicExch;
        REQUIRE(fixture(kernelAtomicExch, value));

        AtomicTestKernel<Min, TAcc, T> kernelAtomicMin;
        REQUIRE(fixture(kernelAtomicMin, value));

        AtomicTestKernel<Max, TAcc, T> kernelAtomicMax;
        REQUIRE(fixture(kernelAtomicMax, value));

        AtomicTestKernel<Inc, TAcc, T> kernelAtomicInc;
        REQUIRE(fixture(kernelAtomicInc, value));

        AtomicTestKernel<Dec, TAcc, T> kernelAtomicDec;
        REQUIRE(fixture(kernelAtomicDec, value));

        AtomicTestKernel<And, TAcc, T> kernelAtomicAnd;
        REQUIRE(fixture(kernelAtomicAnd, value));

        AtomicTestKernel<Or, TAcc, T> kernelAtomicOr;
        REQUIRE(fixture(kernelAtomicOr, value));

        AtomicTestKernel<Xor, TAcc, T> kernelAtomicXor;
        REQUIRE(fixture(kernelAtomicXor, value));
#endif
        AtomicTestKernel<Cas, TAcc, T> kernelAtomicCas;
        REQUIRE(fixture(kernelAtomicCas, value));



    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("atomicOperationsWorking", "[atomic]", TestAccs)
{
    using Acc = TestType;

    TestAtomicOperations<Acc, unsigned char>::testAtomicOperations();
    TestAtomicOperations<Acc, char>::testAtomicOperations();
    TestAtomicOperations<Acc, unsigned short>::testAtomicOperations();
    TestAtomicOperations<Acc, short>::testAtomicOperations();

    TestAtomicOperations<Acc, unsigned int>::testAtomicOperations();
    TestAtomicOperations<Acc, int>::testAtomicOperations();

    TestAtomicOperations<Acc, unsigned long>::testAtomicOperations();
    TestAtomicOperations<Acc, long>::testAtomicOperations();
    TestAtomicOperations<Acc, unsigned long long>::testAtomicOperations();
    TestAtomicOperations<Acc, long long>::testAtomicOperations();

    TestAtomicOperations<Acc, float>::testAtomicOperations();
    TestAtomicOperations<Acc, double>::testAtomicOperations();
}
