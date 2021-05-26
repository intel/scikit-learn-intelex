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

cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(TARGET "dpc_backend")

if(NOT DEFINED ENV{DPCPPROOT})
    message(FATAL_ERROR "Not set DPCPPROOT env")
else()
    set(DPCPPROOT $ENV{DPCPPROOT})
endif()

if(WIN32)
    set(CMAKE_CXX_COMPILER "dpcpp")
    set(CMAKE_C_COMPILER "clang-cl")
    message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
    message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})
    set(WARNING_FLAGS "-Winit-self -Wmissing-declarations -Wno-null-conversion -Wno-unknown-pragmas -Wno-unused-const-variable")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} -Qstd=c++17 /EHsc /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG -Qstd=c++17")
elseif(UNIX)
    set(CMAKE_CXX_COMPILER "dpcpp")
    set(CMAKE_C_COMPILER "clang")
    message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
    message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})
    set(SDL_FLAGS "-fstack-protector -fstack-protector-all -fPIC -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fno-strict-overflow -fno-delete-null-pointer-checks")
    set(WARNING_FLAGS "-Winit-self -Wmissing-declarations -Wno-null-conversion")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS} -std=c++17 -fsycl")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG -std=c++17 -fsycl")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

project("dpcpp part python onedal"
    HOMEPAGE_URL https://github.com/oneapi-src/oneDAL
)

set(PYTHON_ONEDAL "${CMAKE_CURRENT_SOURCE_DIR}/../onedal/")

file(GLOB_RECURSE sources
    ${PYTHON_ONEDAL}/**/*.cpp
)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

set(USE_DPCPP "yes")
set(USE_NEW_IFACES "yes")
set(LINK_TYPE "so")
set(THREADING_TYPE "parallel")

# find_package(oneDAL REQUIRED)
message(STATUS "oneDAL_INCLUDE_DIRS:" ${oneDAL_INCLUDE_DIRS})
message(STATUS "PYTHON_INCLUDE_DIR:" ${PYTHON_INCLUDE_DIR})
message(STATUS "PYTHON_LIBRARY_DIR:" ${PYTHON_LIBRARY_DIR})
message(STATUS "NUMPY_INCLUDE_DIRS:" ${NUMPY_INCLUDE_DIRS})
message(STATUS "DPCTL_INCLUDE_DIR:" ${DPCTL_INCLUDE_DIR})
message(STATUS "DPCTL_LIB_DIR:" ${DPCTL_LIB_DIR})

add_library(${TARGET}
    SHARED
    ${sources}
)

target_include_directories(${TARGET}
    PRIVATE ${PYTHON_ONEDAL}
    PRIVATE ${oneDAL_INCLUDE_DIRS}
    PRIVATE ${PYTHON_INCLUDE_DIR}
    PRIVATE ${NUMPY_INCLUDE_DIRS}
)

target_compile_definitions(${TARGET}
    PUBLIC "ONEDAL_DATA_PARALLEL"
    PUBLIC "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
)

target_link_libraries(${TARGET}
    PRIVATE ${DPCPP_OPENCL_LIBRARY}
)

target_link_directories(${TARGET}
    PRIVATE ${PYTHON_LIBRARY_DIR}
)

# TODO: Get from findOneDAL
if(WIN32)
    set(ONEDAL_LIBRARIES "onedal_core_dll")
    list(APPEND ONEDAL_LIBRARIES "onedal_dpc_dll")
    target_link_libraries(${TARGET} PUBLIC ${ONEDAL_LIBRARIES})

else()
    set(ONEDAL_LIBRARIES "onedal_core")
    list(APPEND ONEDAL_LIBRARIES "onedal_thread")
    list(APPEND ONEDAL_LIBRARIES "onedal_dpc")
    target_link_libraries(${TARGET} PUBLIC ${ONEDAL_LIBRARIES})
endif()

if(DPCTL_ENABLE)
    target_include_directories(${TARGET} PUBLIC ${DPCTL_INCLUDE_DIR})
    target_link_directories(${TARGET} PUBLIC ${DPCTL_LIB_DIR})
    target_link_libraries(${TARGET} PUBLIC "DPCTLSyclInterface")
    target_compile_definitions(${TARGET} PUBLIC "DPCTL_ENABLE")

    # not sure but add runpath
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${DPCTL_LIB_DIR}")

    # disable stripping rpath in installation logic
    set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

install(
  TARGETS ${TARGET}
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)
