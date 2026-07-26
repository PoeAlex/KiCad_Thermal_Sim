cmake_minimum_required(VERSION 3.20)
project(thermalsim_core LANGUAGES CXX)

add_library(thermalsim_core SHARED thermalsim_core.cpp)
target_compile_features(thermalsim_core PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(thermalsim_core PRIVATE /O2 /arch:AVX2 /fp:precise)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(thermalsim_core PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(thermalsim_core PROPERTIES
    OUTPUT_NAME "thermalsim_core"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
