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

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

SET (CMAKE_C_COMPILER_WORKS 1)
SET (CMAKE_CXX_COMPILER_WORKS 1)

project("oneDAL python interface"
    LANGUAGES CXX
    HOMEPAGE_URL https://github.com/oneapi-src/oneDAL
)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    set(SDL_FLAGS "-GS -DynamicBase -fsycl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MD")
elseif(UNIX)
    set(SDL_FLAGS "-fsycl -fstack-protector-strong -fPIC -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fno-strict-overflow -fno-delete-null-pointer-checks")
    if(APPLE)
        set(ONEDAL_PY_LINK_OPTIONS "-fwrapv")
        set(CMAKE_INSTALL_RPATH "@loader_path/../../../")
    else()
        set(ONEDAL_PY_LINK_OPTIONS "-Wl,-z,noexecstack,-z,relro,-z,now,-fstack-protector-strong,-fno-strict-overflow,-fno-delete-null-pointer-checks,-fwrapv")
        set(CMAKE_INSTALL_RPATH "$ORIGIN/../../../")
    endif()
    set(WARNING_FLAGS "-Winit-self")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")

set(PYTHON_ONEDAL "${CMAKE_CURRENT_SOURCE_DIR}/../onedal/")

file(GLOB_RECURSE sources
    ${PYTHON_ONEDAL}/**/*.cpp
    ${PYTHON_ONEDAL}/*.cpp
)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

# find_package(oneDAL REQUIRED)
message(STATUS "oneDAL_INCLUDE_DIRS:" ${oneDAL_INCLUDE_DIRS})
message(STATUS "oneDAL_LIBRARY_DIR:" ${oneDAL_LIBRARY_DIR})
message(STATUS "PYTHON_INCLUDE_DIR:" ${PYTHON_INCLUDE_DIR})
message(STATUS "PYTHON_LIBRARY_DIR:" ${PYTHON_LIBRARY_DIR})
message(STATUS "NUMPY_INCLUDE_DIRS:" ${NUMPY_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)

if(IFACE STREQUAL "host")
    set(TARGET "_onedal_py_host")

    if(WIN32)
        set(ONEDAL_LIBRARIES "onedal_dll" "onedal_core_dll")
    else()
        set(ONEDAL_LIBRARIES "onedal" "onedal_core" "onedal_thread")
    endif()

    list(APPEND COMPILE_DEFINITIONS "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

elseif(IFACE STREQUAL "dpc")
    set(TARGET "_onedal_py_dpc")

    if(WIN32)
        set(ONEDAL_LIBRARIES "onedal_dpc_dll" "onedal_core_dll")
    else()
        set(ONEDAL_LIBRARIES "onedal_dpc" "onedal_core" "onedal_thread")
    endif()

    list(APPEND COMPILE_DEFINITIONS
        "ONEDAL_DATA_PARALLEL"
        "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
else()
    message(FATAL_ERROR "Invalid IFACE value: " ${IFACE})
endif()

pybind11_add_module(${TARGET} ${sources})

target_include_directories(${TARGET}
    PRIVATE ${PYTHON_ONEDAL}../
    PRIVATE ${oneDAL_INCLUDE_DIRS}
    PRIVATE ${PYTHON_INCLUDE_DIR}
    PRIVATE ${NUMPY_INCLUDE_DIRS}
)

target_compile_definitions(${TARGET} PUBLIC ${COMPILE_DEFINITIONS})
target_link_directories(${TARGET} PRIVATE ${PYTHON_LIBRARY_DIR} ${oneDAL_LIBRARY_DIR})
target_link_libraries(${TARGET} PUBLIC ${ONEDAL_LIBRARIES})
target_link_options(${TARGET} PRIVATE ${ONEDAL_PY_LINK_OPTIONS})

install(
  TARGETS ${TARGET}
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)
