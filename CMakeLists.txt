cmake_minimum_required(VERSION 3.24)

project(VRAMBridge VERSION 0.1.0 LANGUAGES CXX)

option(VRAMBRIDGE_ENABLE_CUDA "Build CUDA hardware and bandwidth probes" ON)
option(VRAMBRIDGE_BUILD_TESTS "Build VRAMBridge tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(VRAMBRIDGE_CORE_SOURCES src/info.cpp)

if(VRAMBRIDGE_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
endif()

if(VRAMBRIDGE_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND VRAMBRIDGE_CORE_SOURCES src/cuda/cuda_info.cu)
else()
    list(APPEND VRAMBRIDGE_CORE_SOURCES src/cuda_stub.cpp)
endif()

add_library(vrambridge_core ${VRAMBRIDGE_CORE_SOURCES})
target_include_directories(vrambridge_core PUBLIC include)
target_compile_features(vrambridge_core PUBLIC cxx_std_20)

if(VRAMBRIDGE_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set_target_properties(vrambridge_core PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(vrambridge_core PUBLIC CUDA::cudart)
    target_compile_definitions(vrambridge_core PUBLIC VRAMBRIDGE_HAS_CUDA=1)
endif()

add_executable(vrambridge-info tools/vrambridge-info/main.cpp)
target_link_libraries(vrambridge-info PRIVATE vrambridge_core)

if(VRAMBRIDGE_BUILD_TESTS)
    include(CTest)
endif()

install(TARGETS vrambridge_core vrambridge-info
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(PROGRAMS tools/benchmark/vrambridge-benchmark.py DESTINATION bin RENAME vrambridge-benchmark)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_test(NAME vrambridge-info-json
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_info_json.py $<TARGET_FILE:vrambridge-info>
    )
    add_test(NAME vrambridge-benchmark-harness
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark_harness.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/vrambridge-benchmark.py
    )
endif()