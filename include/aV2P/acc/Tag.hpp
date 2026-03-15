/* Copyright 2025 Simeon Ehrig, Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "aV2P/core/Config.hpp"

#include <iostream>
#include <tuple>
#include <type_traits>

namespace aV2P
{
    struct InterfaceTag
    {
    };
} // namespace aV2P

#define CREATE_ACC_TAG(tag_name)                                                                                      \
    struct tag_name : public aV2P::InterfaceTag                                                                       \
    {                                                                                                                 \
        static std::string get_name()                                                                                 \
        {                                                                                                             \
            return #tag_name;                                                                                         \
        }                                                                                                             \
    }

namespace aV2P
{
    CREATE_ACC_TAG(TagCpuSerial);

    CREATE_ACC_TAG(TagAlpaka3Cpu);
    CREATE_ACC_TAG(TagAlpaka3Omp);

    namespace concepts
    {
        template<typename T>
        concept Tag = requires {
            {
                T::get_name()
            } -> std::same_as<std::string>;
            requires std::default_initializable<T>;
            requires std::derived_from<T, aV2P::InterfaceTag>;
        };
    } // namespace concepts

    namespace trait
    {
        template<typename TAcc>
        struct AccToTag;

        template<concepts::Tag TTag, typename TDim, typename TIdx>
        struct TagToAcc;
    } // namespace trait

    //! \brief maps an acc type to a tag type
    //! \tparam TAcc alpaka acc type
    template<typename TAcc>
    using AccToTag = typename trait::AccToTag<TAcc>::type;

    //! \brief maps a tag type to an acc type
    //! \tparam TTag alpaka tag type
    //! \tparam TDim dimension of the mapped acc type
    //! \tparam TIdx index type of the mapped acc type
    template<concepts::Tag TTag, typename TDim, typename TIdx>
    using TagToAcc = typename trait::TagToAcc<TTag, TDim, TIdx>::type;

    template<typename TAcc, concepts::Tag... TTag>
    inline constexpr bool accMatchesTags = (std::is_same_v<aV2P::AccToTag<TAcc>, TTag> || ...);

    //! list of all available tags
    using AccTags = std::tuple<aV2P::TagCpuSerial, aV2P::TagAlpaka3Cpu, TagAlpaka3Omp>;

    //!  \brief Function to print the names of each tag in the given tuple of tags
    //!  \tparam TTuple is the type of the tuple of tags
    template<typename TTuple>
    void printTagNames()
    {
        // Check if the tuple is empty using std::tuple_size_v
        if(std::tuple_size_v<TTuple> == 0)
        {
            std::cout << "No Tags!";
        }
        else
        {
            std::cout << "Tags: ";
            // Print tags with comma in between
            std::apply(
                [](auto... args)
                {
                    auto index = std::tuple_size_v<TTuple>;
                    ((std::cout << args.get_name() << (--index > 0u ? "," : "")), ...);
                },
                TTuple{});
        }
        std::cout << std::endl;
    }
} // namespace aV2P
