cmake_minimum_required(VERSION 3.18)
project(popbrain_llama_cpp VERSION 1.3.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(POPBRAIN_BUILD_TESTS "Build PopBrain test suite" ON)
option(POPBRAIN_ENABLE_CUDA "Enable CUDA acceleration in PopBrain" OFF)

include_directories(include)

# Shared / Static PopBrain Synapse Library
add_library(popbrain_synapse SHARED
    src/popbrain_synapse.cpp
)

target_compile_definitions(popbrain_synapse PRIVATE POPBRAIN_BUILD_SHARED)

if (POPBRAIN_ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(popbrain_synapse PUBLIC POPBRAIN_USE_CUDA)
    set_target_properties(popbrain_synapse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if (MSVC)
    target_compile_options(popbrain_synapse PRIVATE /W3 /O2 /EHsc)
else()
    target_compile_options(popbrain_synapse PRIVATE -Wall -O3 -fPIC)
endif()

# Test Suite Executable
if (POPBRAIN_BUILD_TESTS)
    add_executable(test_popbrain_synapse
        tests/test_popbrain_synapse.cpp
    )
    target_link_libraries(test_popbrain_synapse PRIVATE popbrain_synapse)
    enable_testing()
    add_test(NAME SynapticMemoryCoreTest COMMAND test_popbrain_synapse)
endif()

# Native PopBrain CLI Management Executable
add_executable(popbrain_cli
    src/popbrain_cli.cpp
)
target_link_libraries(popbrain_cli PRIVATE popbrain_synapse)

# Installation Targets
install(TARGETS popbrain_synapse popbrain_cli
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/popbrain_synapse.h DESTINATION include)
