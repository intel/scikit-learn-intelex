#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO: Get from findOneDAL
if(WIN32)
    set(ONEDAL_LIBRARIES "onedal_core_dll")
else()
    set(ONEDAL_LIBRARIES "onedal_core")
    list(APPEND ONEDAL_LIBRARIES "onedal_thread")
endif()

if(IFACE STREQUAL "host")
    set(TARGET "_onedal_py_host")

    if(WIN32)
        list(APPEND ONEDAL_LIBRARIES "onedal_dll")
    else()
        list(APPEND ONEDAL_LIBRARIES "onedal")
    endif()

elseif(IFACE STREQUAL "dpc")
    if(NOT DEFINED ENV{DPCPPROOT})
        message(FATAL_ERROR "DPCPPROOT is not set")
    endif()

    set(CMAKE_CXX_COMPILER "dpcpp")
    set(CMAKE_C_COMPILER "dpcpp")
    message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
    message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})

    set(TARGET "_onedal_py_dpc")

    if(WIN32)
        list(APPEND ONEDAL_LIBRARIES "onedal_dpc_dll")
    else()
        list(APPEND ONEDAL_LIBRARIES "onedal_dpc")
    endif()
else()
    message(FATAL_ERROR "Invalid IFACE value: " ${IFACE})
endif()

if(WIN32)
    set(WARNING_FLAGS "-Winit-self -Wmissing-declarations -Wno-null-conversion -Wno-unknown-pragmas -Wno-unused-const-variable")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} /EHsc /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
elseif(UNIX OR APPLE)
    set(SDL_FLAGS "-fstack-protector -fstack-protector-all -fPIC -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fno-strict-overflow -fno-delete-null-pointer-checks")
    set(WARNING_FLAGS "-Winit-self -Wmissing-declarations -Wno-null-conversion")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")

    # if(APPLE)
    #     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -undefined dynamic_lookup")
    #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
    #     set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -undefined dynamic_lookup")
    #     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -undefined dynamic_lookup")
    # endif()
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

project("oneDAL python interface"
    HOMEPAGE_URL https://github.com/oneapi-src/oneDAL
)

set(PYTHON_ONEDAL "${CMAKE_CURRENT_SOURCE_DIR}/../onedal/")

file(GLOB_RECURSE sources
    ${PYTHON_ONEDAL}/**/*.cpp
    ${PYTHON_ONEDAL}/*.cpp
)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

# find_package(oneDAL REQUIRED)
message(STATUS "oneDAL_INCLUDE_DIRS:" ${oneDAL_INCLUDE_DIRS})
message(STATUS "PYTHON_INCLUDE_DIR:" ${PYTHON_INCLUDE_DIR})
message(STATUS "PYTHON_LIBRARY_DIR:" ${PYTHON_LIBRARY_DIR})
message(STATUS "NUMPY_INCLUDE_DIRS:" ${NUMPY_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)

pybind11_add_module(${TARGET} ${sources})

target_include_directories(${TARGET}
    PRIVATE ${PYTHON_ONEDAL}../
    PRIVATE ${oneDAL_INCLUDE_DIRS}
    PRIVATE ${PYTHON_INCLUDE_DIR}
    PRIVATE ${NUMPY_INCLUDE_DIRS}
)

if(IFACE STREQUAL "host")
    target_compile_definitions(${TARGET}
        PUBLIC "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
    )
elseif(IFACE STREQUAL "dpc")
    target_compile_definitions(${TARGET}
        PUBLIC "ONEDAL_DATA_PARALLEL"
        PUBLIC "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
    )
endif()

target_link_directories(${TARGET}
    PRIVATE ${PYTHON_LIBRARY_DIR}
)

target_link_libraries(${TARGET} PUBLIC ${ONEDAL_LIBRARIES})

install(
  TARGETS ${TARGET}
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)
