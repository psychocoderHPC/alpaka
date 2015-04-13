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

#pragma once

#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST

#include <boost/algorithm/string/split.hpp> // boost::algorithm::split
#include <boost/predef.h>                   // Workarounds

#include <iostream>                         // std::cerr
#include <string>                           // std::string

/*#if BOOST_OS_WINDOWS
    #ifdef ALPAKA_CUDA_ENABLED
        #include <cuda.h>                   // CUDA driver API

        #if (!defined(CUDA_VERSION) || (CUDA_VERSION < 7000))
            #error "CUDA version 7.0 or greater required!"
        #endif

        namespace alpaka
        {
            namespace hwloc
            {
                namespace cuda
                {
                    namespace detail
                    {
                        //-----------------------------------------------------------------------------
                        //! CUDA runtime error checking with log and exception, ignoring specific error values
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto cudaDrvCheck(
                            CUresult const error,
                            std::string const cmd,
                            std::string const file,
                            std::size_t const line)
                        -> void
                        {
                            // Even if we get the error directly from the command, we have to reset the global error state by getting it.
                            if(error != CUDA_SUCCESS)
                            {
                                std::string const sError(file + "(" + std::to_string(line) + ") '" + cmd + "' returned error: '" + std::to_string(error) + "' (possibly from a previous CUDA call)!");
                                std::cerr << sError << std::endl;
                                ALPAKA_DEBUG_BREAK;
                                throw std::runtime_error(sError);
                            }
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------
        //! CUDA driver error checking with log and exception.
        //-----------------------------------------------------------------------------
        #define ALPAKA_CUDA_DRV_CHECK(cmd)\
            ::alpaka::hwloc::cuda::detail::cudaDrvCheck(cmd, #cmd, __FILE__, __LINE__)
    #endif
#endif*/

#if BOOST_COMP_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4100)  // hwloc\include\hwloc/helper.h(227): warning C4100: 'topology': unreferenced formal parameter
    #pragma warning(disable: 4127)  // hwloc\include\hwloc/helper.h(288): warning C4127: conditional expression is constant
    #pragma warning(disable: 4996)  // hwloc\include\hwloc/helper.h(1218): warning C4996: 'sscanf': This function or variable may be unsafe.
#endif

#include <hwloc.h>

#if BOOST_COMP_MSVC
    #pragma warning(pop)
#endif

#if HWLOC_API_VERSION < 0x00010300
    #error "hwloc version 1.3 or greater required!"
#endif



namespace alpaka
{
    namespace hwloc
    {
        //-----------------------------------------------------------------------------
        //! hwloc runtime error checking with log and exception, ignoring specific error values
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST auto hwlocCheck(
            int const error,
            std::string const cmd,
            std::string const file,
            std::size_t const line)
        -> void
        {
            // Even if we get the error directly from the command, we have to reset the global error state by getting it.
            if(error != 0)
            {
                std::string const sError(file + "(" + std::to_string(line) + ") '" + cmd + "' returned error: '" + std::to_string(error) + "' (possibly from a previous CUDA call)!");
                std::cerr << sError << std::endl;
                ALPAKA_DEBUG_BREAK;
                throw std::runtime_error(sError);
            }
        }

        //-----------------------------------------------------------------------------
        //! hwloc error checking.
        //-----------------------------------------------------------------------------
        #define ALPAKA_HWLOC_CHECK(cmd)\
            ::alpaka::hwloc::hwlocCheck(cmd, #cmd, __FILE__, __LINE__)

        //#############################################################################
        //! The hardware topology of the current process.
        //#############################################################################
        class HardwareTopology
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST HardwareTopology()
            {
                // Allocate and initialize the topology.
                ALPAKA_HWLOC_CHECK(hwloc_topology_init(&m_Topology));
                // Enable the detection of PCI devices.
                // hwloc pci backend requires: HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM && (HWLOC_TOPOLOGY_FLAG_WHOLE_IO || HWLOC_TOPOLOGY_FLAG_IO_DEVICES)
                // hwloc cuda backend requires: HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM && (HWLOC_TOPOLOGY_FLAG_WHOLE_IO || HWLOC_TOPOLOGY_FLAG_IO_DEVICES)
                hwloc_topology_set_flags(
                    m_Topology,
                    HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM |
                    HWLOC_TOPOLOGY_FLAG_IO_DEVICES | HWLOC_TOPOLOGY_FLAG_IO_BRIDGES | HWLOC_TOPOLOGY_FLAG_WHOLE_IO |
                    HWLOC_TOPOLOGY_FLAG_ICACHES |
                    HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM | // hwloc cuda backend requires this to be set
                    hwloc_topology_get_flags(m_Topology));
                // Perform the topology detection.
                ALPAKA_HWLOC_CHECK(hwloc_topology_load(m_Topology));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                /*int const iTopoDepth(hwloc_topology_get_depth(m_Topology));
                std::cout
                    << "Topology depth: " << iTopoDepth << std::endl
                    << std::endl;*/

                // Walk the topology with an array style, from level 0 (always the system level) to the lowest level (always the proc level).
                /*char string[128];
                for(int iDepth(0); iDepth < iTopoDepth; ++iDepth)
                {
                    std::cout << "***  Objects at level " << iDepth << std::endl;
                    int const iNumObjectsAtDepth(hwloc_get_nbobjs_by_depth(m_Topology, iDepth));
                    for(int i(0); i < iNumObjectsAtDepth; ++i)
                    {
                        hwloc_obj_snprintf(string, sizeof(string), m_Topology, hwloc_get_obj_by_depth(m_Topology, iDepth, i), "#", m_iVerbose);
                        std::cout << "Index " << i << ": "<< string << std::endl;
                    }
                }
                std::cout << std::endl;*/

                // Compute the amount of cache that the first logical processor has above it.
                /*std::size_t uiCacheLevels(0);
                int iCacheSize(0);
                for (hwloc_obj_t obj = hwloc_get_obj_by_type(m_Topology, HWLOC_OBJ_PU, 0); obj; obj = obj->parent)
                {
                    if(obj->type == HWLOC_OBJ_CACHE)
                    {
                        uiCacheLevels++;
                        iCacheSize += obj->attr->cache.size;
                    }
                }
                std::cout << "*** Logical processor 0 has " << uiCacheLevels << " caches totaling " << iCacheSize / 1024u << " KB!" << std::endl;
                std::cout << std::endl;*/

                // Walk the topology tree.
                printChildren(std::cout, hwloc_get_root_obj(m_Topology), 0);
                std::cout << std::endl;
#endif
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~HardwareTopology()
            {
                hwloc_topology_destroy(m_Topology);
            }

        private:

            //-----------------------------------------------------------------------------
            //! Print the object information.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto printObject(
                std::ostream & os,
                hwloc_obj_t const & obj) const
            -> void
            {
                // Print the type.
                int const iTypeStringLength(hwloc_obj_type_snprintf(nullptr, 0u, obj, m_iVerbose));
                std::string sType(static_cast<std::size_t>(iTypeStringLength+1u), ' ');
                hwloc_obj_type_snprintf(&sType.front(), iTypeStringLength+1, obj, m_iVerbose);
                os << sType;

                // Print its attributes.
                int const iAttrStringLength(hwloc_obj_attr_snprintf (NULL, 0, obj, ", ", m_iVerbose));
                std::string sAttr(static_cast<std::size_t>(iAttrStringLength+1u), ' ');
                hwloc_obj_attr_snprintf(&sAttr.front(), iAttrStringLength+1, obj, ", ", m_iVerbose);
                os << ": " << sAttr << std::endl;
            }
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
            //! Recursively print the object and its children.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto printChildren(
                std::ostream & os,
                hwloc_obj_t const & obj,
                int const & depth) const
            -> void
            {
                // Print the object.
                os << std::string(2*depth, ' ');
                printObject(
                    os,
                    obj);

                // Print its children.
                for(unsigned int i(0); i < obj->arity; i++)
                {
                    printChildren(
                        os,
                        obj->children[i],
                        depth + 1);
                }
            }
#endif
/*#if BOOST_OS_WINDOWS
    #ifdef ALPAKA_CUDA_ENABLED
        //-----------------------------------------------------------------------------
        //! \return The numa node the given CUDA device is connected to.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST auto getCudaDevNumaNode(
            CUdevice const & dev)
        -> long
        {
            // Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values.
            // pciBusId should be large enough to store 13 characters including the NULL-terminator
            std::string sDevPciBusId(13u, ' ');
            ALPAKA_CUDA_DRV_CHECK(cuDeviceGetPCIBusId(
                &sDevPciBusId.front(),
                13,
                dev));

            std::vector<std::string> vsTokens;
            boost::algorithm::split(
                vsTokens,
                sDevPciBusId,
                boost::algorithm::is_any_of(":."));
            assert(vsTokens.size() == 4);

            unsigned long cudaBus(std::stoul(vsTokens[1], nullptr, 16));
            unsigned long cudaSubdevice(std::stoul(vsTokens[2], nullptr, 16));
            unsigned long cudaFunction(std::stoul(vsTokens[3], nullptr, 16));

            // Get a device information set.
            HDEVINFO hNvDevInfo = SetupDiGetClassDevs(
                nullptr,
                nullptr,
                nullptr,
                DIGCF_PRESENT | DIGCF_ALLCLASSES);
            if(hNvDevInfo == INVALID_HANDLE_VALUE)
            {
                // TODO: Log GetLastError()
                throw std::runtime_error("SetupDiGetClassDevs returned INVALID_HANDLE_VALUE");
            }

            // Find the deviceInfoData for each GPU
            DWORD deviceIndex;
            for(deviceIndex = 0; ; deviceIndex++)
            {
                SP_DEVINFO_DATA deviceInfoData;
                deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
                // MSDN: The installer should then increment MemberIndex and call SetupDiEnumDeviceInfo until there are no more values
                // (the function fails and a call to GetLastError returns ERROR_NO_MORE_ITEMS).
                ret = SetupDiEnumDeviceInfo(
                    hNvDevInfo,
                    deviceIndex,
                    &deviceInfoData);
                if(!ret)
                {
                    DWORD const lastError(GetLastError());
                    if(hNvDevInfo == ERROR_NO_MORE_ITEMS)
                    {
                        break;
                    }
                    else
                    {
                        // TODO: Log GetLastError()
                        throw std::runtime_error("SetupDiEnumDeviceInfo failed");
                    }
                }

                // Get the size of the info.
                DWORD iReqSize(0);
                ret = SetupDiGetDeviceRegistryProperty(
                    hNvDevInfo,
                    &deviceInfoData,
                    SPDRP_LOCATION_INFORMATION,
                    nullptr,
                    nullptr,
                    0,
                    &iReqSize);
                std::string sLocInfo(static_cast<std::size_t>(iReqSize), ' ');
                ret = SetupDiGetDeviceRegistryProperty(
                    hNvDevInfo,
                    &deviceInfoData,
                    SPDRP_LOCATION_INFORMATION,
                    nullptr,
                    static_cast<PBYTE>(&sLocInfo.front()),
                    iReqSize,
                    nullptr);
                if(!ret)
                {
                    // TODO: Log GetLastError()
                    throw std::runtime_error("SetupDiGetDeviceRegistryProperty failed");
                }

                unsigned long bus(0u);
                unsigned long subdevice(0u);
                unsigned long function(0u);

                if(sLocInfo.substr(0u, 3u) == "PCI")
                {
                    std::size_t const uiBusPos(sLocInfo.find("bus", 3));
                    if(uiBusPos != string::npos)
                    {
                        std::size_t const uiBusCommaPos(sLocInfo.find(",", uiBusPos));
                        if(uiBusCommaPos != string::npos)
                        {
                            bus = std::stoi(sLocInfo.substr(uiBusPos+3, uiBusCommaPos-(uiBusPos+3)));

                            std::size_t const uiDevPos(sLocInfo.find("device", uiBusCommaPos));
                            if(uiDevPos != string::npos)
                            {
                                std::size_t const uiDevCommaPos(sLocInfo.find(",", uiDevPos));
                                if(uiDevCommaPos != string::npos)
                                {
                                    subdevice = std::stoi(sLocInfo.substr(uiDevPos+6, uiDevCommaPos-(uiDevPos+6)));

                                    std::size_t const uiFuncPos(sLocInfo.find("function", uiDevCommaPos));
                                    if(uiFuncPos != string::npos)
                                    {
                                        subdevice = std::stoi(sLocInfo.substr(uiFuncPos+8));
                                    }
                                }
                                else
                                {
                                    throw std::runtime_error("Error while parsing SPDRP_LOCATION_INFORMATION: 'device'...','");
                                }
                            }
                            else
                            {
                                throw std::runtime_error("Error while parsing SPDRP_LOCATION_INFORMATION: 'device'");
                            }
                        }
                        else
                        {
                            throw std::runtime_error("Error while parsing SPDRP_LOCATION_INFORMATION: 'bus'...','");
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Error while parsing SPDRP_LOCATION_INFORMATION: 'bus'");
                    }
                }
                else
                {
                    throw std::runtime_error("Error while parsing SPDRP_LOCATION_INFORMATION: 'PCI'");
                }

                // If the PCI location matches the one from CUDA we have found our device.
                if( (bus == cudaBus) &&
                    (subdevice == cudaSubdevice) &&
                    (function == cudaFunction))
                {
                    ret = SetupDiGetDeviceRegistryProperty(hNvDevInfo, &deviceInfoData, SPDRP_HARDWAREID, NULL,
                        (PBYTE)locinfo, sizeof(locinfo), NULL);
                    printf("locinfo %s\n", locinfo);
                    int data[20];
                    data[0] = 0;
                    DEVPROPTYPE type;
                    DEVPROPKEY key = DEVPKEY_Numa_Proximity_Domain;
                    lastError = 0;
                    ret =  SetupDiGetDeviceProperty(hNvDevInfo, &deviceInfoData,&key , &type, (PBYTE)&data[0], 20*sizeof(int), NULL,0);
                    if(!ret)
                    {
                        lastError = GetLastError();
                    }
                    printf("DEVPKEY_Numa_Proximity_Domain %d err %d\n", data[0], lastError);
                    key = DEVPKEY_Device_Numa_Node;
                    lastError = 0;
                    ret =  SetupDiGetDeviceProperty(hNvDevInfo, &deviceInfoData,&key , &type, (PBYTE)&data[0], 20*sizeof(int), NULL,0);
                    if(!ret)
                    {
                        lastError = GetLastError();
                    }
                    printf("DEVPKEY_Device_Numa_Node %d err %d\n", data[0], lastError);
                    return data[0];
                }
            }
            return -1;
        }
    #endif
#endif*/

        public:
            hwloc_topology_t m_Topology;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            static int const m_iVerbose = 1;
#else
            static int const m_iVerbose = 0;
#endif
        };

        //! The topology information as seen from the current process.
        static HardwareTopology g_HardwareTopology;
    }
}