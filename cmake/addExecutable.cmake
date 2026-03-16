#
# Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

#------------------------------------------------------------------------------
#
# alpaka_add_executable(<name> [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [<source>...])
#
# Calls add_executable under the hood. Depending on the enabled back-ends, source file or target properties which
# cannot be propagated by the alpaka::alpaka target are set here.
#
# Using a macro to stay in the scope (fixes lost assignment of linker command in FindHIP.cmake)
# https://github.com/ROCm-Developer-Tools/HIP/issues/631

macro(alpaka_add_executable In_Name)

    add_executable(${In_Name} ${ARGN})

    target_link_libraries(${In_Name} PUBLIC alpaka3::alpaka)
    alpaka_finalize(${In_Name})
endmacro()
