cmake_minimum_required(VERSION 3.24)

project(VRAMBridge VERSION 0.1.0 LANGUAGES CXX)

option(VRAMBRIDGE_ENABLE_CUDA "Build CUDA hardware and bandwidth probes" ON)
option(VRAMBRIDGE_BUILD_TESTS "Build VRAMBridge tests" ON)
set(VRAMBRIDGE_LLAMA_SOURCE "" CACHE PATH "Pinned llama.cpp source for the optional routing collector")
set(VRAMBRIDGE_LLAMA_LIBRARY_DIR "" CACHE PATH "Matching official llama.cpp shared libraries")
option(VRAMBRIDGE_TEST_MASKED_IDS "Test the opt-in patched CPU masked-expert primitive (requires patched libraries)" OFF)

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
find_package(Threads REQUIRED)
target_link_libraries(vrambridge_core PRIVATE Threads::Threads)
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

if(VRAMBRIDGE_LLAMA_SOURCE OR VRAMBRIDGE_LLAMA_LIBRARY_DIR)
    if(NOT VRAMBRIDGE_LLAMA_SOURCE OR NOT VRAMBRIDGE_LLAMA_LIBRARY_DIR)
        message(FATAL_ERROR "Routing collection requires both llama.cpp source and matching library paths")
    endif()
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${VRAMBRIDGE_LLAMA_SOURCE} rev-parse HEAD
        OUTPUT_VARIABLE VRAMBRIDGE_LLAMA_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE VRAMBRIDGE_LLAMA_GIT_RESULT
    )
    if(NOT VRAMBRIDGE_LLAMA_GIT_RESULT EQUAL 0 OR
       NOT VRAMBRIDGE_LLAMA_COMMIT STREQUAL "620d2225cc436ab0b001e3200e590e3b286ef9ab")
        message(FATAL_ERROR "Routing collector currently requires official llama.cpp commit 620d2225cc436ab0b001e3200e590e3b286ef9ab")
    endif()
    foreach(LIBRARY IN ITEMS llama llama-common ggml ggml-base)
        string(REPLACE "-" "_" VARIABLE ${LIBRARY})
        unset(VRAMBRIDGE_${VARIABLE}_LIBRARY CACHE)
        find_library(VRAMBRIDGE_${VARIABLE}_LIBRARY NAMES ${LIBRARY}
            PATHS ${VRAMBRIDGE_LLAMA_LIBRARY_DIR} NO_DEFAULT_PATH NO_CACHE REQUIRED)
        list(APPEND VRAMBRIDGE_TRACE_LIBRARIES ${VRAMBRIDGE_${VARIABLE}_LIBRARY})
    endforeach()
    find_package(Threads REQUIRED)
    add_executable(vrambridge-trace tools/trace/routing_trace.cpp)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/routing_trace.cpp)
    file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/routing_trace.cpp VRAMBRIDGE_TRACE_SOURCE_SHA256)
    target_include_directories(vrambridge-trace PRIVATE
        ${VRAMBRIDGE_LLAMA_SOURCE}/include
        ${VRAMBRIDGE_LLAMA_SOURCE}/common
        ${VRAMBRIDGE_LLAMA_SOURCE}/src
        ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include
        ${VRAMBRIDGE_LLAMA_SOURCE}/vendor
    )
    target_link_libraries(vrambridge-trace PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES} Threads::Threads)
    target_compile_definitions(vrambridge-trace PRIVATE
        VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}"
        VRAMBRIDGE_TRACE_SOURCE_SHA256="${VRAMBRIDGE_TRACE_SOURCE_SHA256}")
    add_executable(vrambridge-hybrid-layer tools/trace/hybrid_layer.cpp)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/hybrid_layer.cpp)
    file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/hybrid_layer.cpp VRAMBRIDGE_HYBRID_SOURCE_SHA256)
    target_include_directories(vrambridge-hybrid-layer PRIVATE
        ${VRAMBRIDGE_LLAMA_SOURCE}/common
        ${VRAMBRIDGE_LLAMA_SOURCE}/include
        ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include
        ${VRAMBRIDGE_LLAMA_SOURCE}/vendor)
    target_link_libraries(vrambridge-hybrid-layer PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES} Threads::Threads)
    target_compile_definitions(vrambridge-hybrid-layer PRIVATE
        VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}"
        VRAMBRIDGE_HYBRID_SOURCE_SHA256="${VRAMBRIDGE_HYBRID_SOURCE_SHA256}")
    if(VRAMBRIDGE_TEST_MASKED_IDS)
        add_executable(vrambridge-admission-test tests/test_moe_admission.cpp)
        target_include_directories(vrambridge-admission-test PRIVATE
            ${VRAMBRIDGE_LLAMA_SOURCE}/src ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include)
        add_executable(vrambridge-masked-matmul-test tests/test_masked_matmul.cpp)
        target_include_directories(vrambridge-masked-matmul-test PRIVATE ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include)
        target_link_libraries(vrambridge-masked-matmul-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES})
        target_compile_definitions(vrambridge-masked-matmul-test PRIVATE
            VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}")
        add_executable(vrambridge-logit-check tools/trace/logit_check.cpp)
        get_target_property(VRAMBRIDGE_TRACE_INCLUDES vrambridge-trace INCLUDE_DIRECTORIES)
        target_include_directories(vrambridge-logit-check PRIVATE ${VRAMBRIDGE_TRACE_INCLUDES})
        target_link_libraries(vrambridge-logit-check PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES} Threads::Threads)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/logit_check.cpp)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/logit_check.cpp VRAMBRIDGE_LOGIT_SOURCE_SHA256)
        target_compile_definitions(vrambridge-logit-check PRIVATE
            VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}"
            VRAMBRIDGE_LOGIT_SOURCE_SHA256="${VRAMBRIDGE_LOGIT_SOURCE_SHA256}")
    endif()
endif()

if(VRAMBRIDGE_BUILD_TESTS)
    include(CTest)
endif()

install(TARGETS vrambridge_core vrambridge-info
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(PROGRAMS tools/benchmark/vrambridge-benchmark.py DESTINATION bin RENAME vrambridge-benchmark)
install(PROGRAMS tools/benchmark/vrambridge-routing.py DESTINATION bin RENAME vrambridge-routing)
install(PROGRAMS tools/benchmark/vrambridge-execution-sweep.py DESTINATION bin RENAME vrambridge-execution-sweep)
install(PROGRAMS tools/vrambridge/app.py DESTINATION share/vrambridge)
install(FILES tools/vrambridge/hybrid-profile.example.json DESTINATION share/vrambridge)
install(DIRECTORY tools/vrambridge/web DESTINATION share/vrambridge)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_test(NAME vrambridge-info-json
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_info_json.py $<TARGET_FILE:vrambridge-info>
    )
    add_test(NAME vrambridge-benchmark-harness
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark_harness.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/vrambridge-benchmark.py
    )
    add_test(NAME vrambridge-local-app
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_local_app.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/vrambridge/app.py
    )
    add_test(NAME vrambridge-routing-analysis
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_routing_analysis.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/vrambridge-routing.py
    )
    add_test(NAME vrambridge-execution-sweep
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_execution_sweep.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/vrambridge-execution-sweep.py
    )
    add_test(NAME vrambridge-model-matrix
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_matrix.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/vrambridge-model-matrix.py
    )
    add_test(NAME vrambridge-moe-fixture-inputs
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_moe_fixtures.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/make_moe_fixture.py
    )
    find_program(NODE_EXECUTABLE node)
    add_test(NAME synapse-telemetry
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_telemetry.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/vrambridge/telemetry.py)
    if(NODE_EXECUTABLE)
        add_test(NAME vrambridge-chat-stream
            COMMAND ${NODE_EXECUTABLE} --test ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_chat_stream.mjs
        )
        add_test(NAME synapse-monitor-navigation
            COMMAND ${NODE_EXECUTABLE} --test ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monitor.mjs)
    else()
        message(STATUS "Node.js not found; browser stream unit tests are unavailable")
    endif()
    if(TARGET vrambridge-trace)
        add_test(NAME vrambridge-trace-help COMMAND vrambridge-trace --help)
        add_test(NAME vrambridge-trace-inputs
            COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_routing_collector.py
                $<TARGET_FILE:vrambridge-trace>
        )
    endif()
    if(TARGET vrambridge-hybrid-layer)
        add_test(NAME vrambridge-hybrid-layer-self-test COMMAND vrambridge-hybrid-layer --self-test --repetitions 3)
        add_test(NAME vrambridge-hybrid-layer-inputs
            COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hybrid_layer.py
                $<TARGET_FILE:vrambridge-hybrid-layer>
        )
    endif()
    if(TARGET vrambridge-masked-matmul-test)
        add_test(NAME vrambridge-moe-admission COMMAND vrambridge-admission-test)
        add_test(NAME vrambridge-masked-matmul COMMAND ${CMAKE_COMMAND} -E env
            VRAMBRIDGE_HYBRID_CPU_EMPTY=0 $<TARGET_FILE:vrambridge-masked-matmul-test>)
        add_test(NAME vrambridge-empty-expert COMMAND ${CMAKE_COMMAND} -E env
            VRAMBRIDGE_HYBRID_CPU_EMPTY=1 $<TARGET_FILE:vrambridge-masked-matmul-test>)
        add_test(NAME vrambridge-shared-prepare-off COMMAND ${CMAKE_COMMAND} -E env
            VRAMBRIDGE_HYBRID_CPU_EMPTY=0 VRAMBRIDGE_HYBRID_CPU_PREPARE=0 $<TARGET_FILE:vrambridge-masked-matmul-test>)
        add_test(NAME vrambridge-shared-prepare-on COMMAND ${CMAKE_COMMAND} -E env
            VRAMBRIDGE_HYBRID_CPU_EMPTY=0 VRAMBRIDGE_HYBRID_CPU_PREPARE=1 $<TARGET_FILE:vrambridge-masked-matmul-test>)
        add_test(NAME vrambridge-logit-check-help COMMAND vrambridge-logit-check --help)
        add_test(NAME vrambridge-logit-check-inputs
            COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_logit_checker.py
                $<TARGET_FILE:vrambridge-logit-check>
        )
    endif()
endif()