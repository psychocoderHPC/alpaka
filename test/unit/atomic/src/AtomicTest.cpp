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
ALPAKA_FN_ACC auto testAtomicCall(TAcc const& acc, bool* success, T& operand, T operandOrig, T value) -> void
{
    auto op = typename TOp::Op{};

    // check if the function `alpaka::atomicOp<*>` is callable
    {
        // left operand is half of the right
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
        // left operand is half of the right
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

ALPAKA_NO_HOST_ACC_WARNING
template<typename THierarchy, typename TOp, typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicCombinations(TAcc const& acc, bool* success, T& operand, T operandOrig) -> void
{
    // helper variables to avoid compiler conversion warnings/errors
    T constexpr one = static_cast<T>(1);
    T constexpr two = static_cast<T>(2);
    {
        // left operand is half of the right
        T const value = static_cast<T>(operandOrig / two);
        testAtomicCall<THierarchy, TOp>(acc, success, operand, operandOrig, value);
    }
    {
        // left operand is twice as large as the right
        T const value = static_cast<T>(operandOrig * two);
        testAtomicCall<THierarchy, TOp>(acc, success, operand, operandOrig, value);
    }
    {
        // left operand is larger by one
        T const value = static_cast<T>(operandOrig + one);
        testAtomicCall<THierarchy, TOp>(acc, success, operand, operandOrig, value);
    }
    {
        // left operand is smaller by one
        T const value = static_cast<T>(operandOrig - one);
        testAtomicCall<THierarchy, TOp>(acc, success, operand, operandOrig, value);
    }
    {
        // both operands are equal
        T const value = operandOrig;
        testAtomicCall<THierarchy, TOp>(acc, success, operand, operandOrig, value);
    }
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename THierarchy, typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicCas(TAcc const& acc, bool* success, T& operand, T operandOrig) -> void
{
    T const value = static_cast<T>(4);

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

//! check threads hierarchy
ALPAKA_NO_HOST_ACC_WARNING
template<typename TOp, typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicHierarchies(TAcc const& acc, bool* success, T& operand, T operandOrig) -> void
{
    testAtomicCombinations<alpaka::hierarchy::Threads, TOp>(acc, success, operand, operandOrig);
    testAtomicCombinations<alpaka::hierarchy::Blocks, TOp>(acc, success, operand, operandOrig);
    testAtomicCombinations<alpaka::hierarchy::Grids, TOp>(acc, success, operand, operandOrig);
}

//! check all alpaka hierarchies
ALPAKA_NO_HOST_ACC_WARNING
template<typename TOp, typename TAcc, typename T>
ALPAKA_FN_ACC auto testAtomicCasHierarchies(TAcc const& acc, bool* success, T& operand, T operandOrig) -> void
{
    testAtomicCas<alpaka::hierarchy::Threads>(acc, success, operand, operandOrig);
    testAtomicCas<alpaka::hierarchy::Blocks>(acc, success, operand, operandOrig);
    testAtomicCas<alpaka::hierarchy::Grids>(acc, success, operand, operandOrig);
}

template<typename TAcc, typename T, typename Sfinae = void>
class AtomicTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T operandOrig) const -> void
    {
        auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);

        testAtomicHierarchies<Add>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Sub>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Exch>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Min>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Max>(acc, success, operand, operandOrig);

        testAtomicHierarchies<And>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Or>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Xor>(acc, success, operand, operandOrig);

        testAtomicHierarchies<Inc>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Dec>(acc, success, operand, operandOrig);

        testAtomicCasHierarchies<alpaka::hierarchy::Threads>(acc, success, operand, operandOrig);
    }
};


template<typename TAcc, typename T>
class AtomicTestKernel<TAcc, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T operandOrig) const -> void
    {
        auto& operand = alpaka::declareSharedVar<T, __COUNTER__>(acc);

        testAtomicHierarchies<Add>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Sub>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Exch>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Min>(acc, success, operand, operandOrig);
        testAtomicHierarchies<Max>(acc, success, operand, operandOrig);

        // Inc, Dec, Or, And, Xor are not supported on float/double types

        testAtomicCasHierarchies<alpaka::hierarchy::Threads>(acc, success, operand, operandOrig);
    }
};


#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)

template<typename TApi, typename TDim, typename TIdx, typename T>
class AtomicTestKernel<
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

template<typename TDim, typename TIdx, typename T>
class AtomicTestKernel<alpaka::AccOacc<TDim, TIdx>, T, std::enable_if_t<sizeof(T) != 4u && sizeof(T) != 8u>>
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

        AtomicTestKernel<TAcc, T> kernel;
        REQUIRE(fixture(kernel, value));
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
