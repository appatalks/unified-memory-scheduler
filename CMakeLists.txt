cmake_minimum_required(VERSION 3.24)

project(VRAMBridge VERSION 0.1.0 LANGUAGES CXX)

option(VRAMBRIDGE_ENABLE_CUDA "Build CUDA hardware and bandwidth probes" ON)
option(VRAMBRIDGE_BUILD_TESTS "Build VRAMBridge tests" ON)
set(VRAMBRIDGE_LLAMA_SOURCE "" CACHE PATH "Pinned llama.cpp source for the optional routing collector")
set(VRAMBRIDGE_LLAMA_LIBRARY_DIR "" CACHE PATH "Matching official llama.cpp shared libraries")
option(VRAMBRIDGE_TEST_MASKED_IDS "Test the opt-in patched CPU masked-expert primitive (requires patched libraries)" OFF)
option(VRAMBRIDGE_TEST_PREFILL_PHASE "Test the separate experimental backend prefill phase API" OFF)
option(VRAMBRIDGE_TEST_COMPATIBLE_DOWN "Build the isolated CPU-compatible MXFP4/Q8 CUDA down probe" OFF)
option(VRAMBRIDGE_CPU_RESEARCH "Build Linux x86 CPU graph profiler and MXFP4 VNNI research shims" OFF)
option(VRAMBRIDGE_CPU_PROFILE_NVTX "Annotate research CPU graph calls for CUDA timeline alignment" OFF)
option(VRAMBRIDGE_TEST_ATTENTION_WINDOW "Build the default-off native Flash Attention query-window research interposer" OFF)
option(VRAMBRIDGE_TEST_EXACT_PACKED_CPU "Build the AVX512 native-contract packed MXFP4 CPU research tests" OFF)
option(VRAMBRIDGE_TEST_KV_COLD_TIER "Build the default-off managed KV cold-page staging experiment" OFF)
set(VRAMBRIDGE_CPU_REFERENCE_LIBRARY "" CACHE FILEPATH "Exact installed CPU plugin for packed-kernel reference tests")

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
    add_executable(vrambridge-render-corpus tools/trace/render_corpus.cpp)
    get_target_property(VRAMBRIDGE_RENDER_INCLUDES vrambridge-trace INCLUDE_DIRECTORIES)
    target_include_directories(vrambridge-render-corpus PRIVATE ${VRAMBRIDGE_RENDER_INCLUDES})
    target_link_libraries(vrambridge-render-corpus PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES} Threads::Threads)
    if(VRAMBRIDGE_TEST_KV_COLD_TIER)
        find_path(VRAMBRIDGE_CUDA12_RUNTIME_INCLUDE cuda_runtime_api.h
            PATHS /usr/local/cuda-12/targets/x86_64-linux/include NO_DEFAULT_PATH REQUIRED)
        find_library(VRAMBRIDGE_CUDA12_RUNTIME NAMES cudart
            PATHS /usr/local/cuda-12/targets/x86_64-linux/lib NO_DEFAULT_PATH REQUIRED)
        add_library(vrambridge-kv-cold-tier SHARED tools/trace/kv_cold_tier.cpp)
        target_include_directories(vrambridge-kv-cold-tier PRIVATE ${VRAMBRIDGE_RENDER_INCLUDES}
            ${VRAMBRIDGE_CUDA12_RUNTIME_INCLUDE})
        target_link_libraries(vrambridge-kv-cold-tier PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES}
            ${VRAMBRIDGE_CUDA12_RUNTIME} Threads::Threads ${CMAKE_DL_LIBS})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/kv_cold_tier.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/kv_cold_tier.h)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/kv_cold_tier.cpp VRAMBRIDGE_KV_COLD_SOURCE_SHA256)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/kv_cold_tier.h VRAMBRIDGE_KV_COLD_HEADER_SHA256)
        target_compile_definitions(vrambridge-kv-cold-tier PRIVATE
            VRAMBRIDGE_KV_CUDA_RUNTIME_LIBRARY="${VRAMBRIDGE_CUDA12_RUNTIME}"
            VRAMBRIDGE_KV_COLD_SOURCE_SHA256="${VRAMBRIDGE_KV_COLD_SOURCE_SHA256}"
            VRAMBRIDGE_KV_COLD_HEADER_SHA256="${VRAMBRIDGE_KV_COLD_HEADER_SHA256}")
        add_executable(vrambridge-kv-cold-tier-test tests/test_kv_cold_tier.cpp)
        target_include_directories(vrambridge-kv-cold-tier-test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace ${VRAMBRIDGE_RENDER_INCLUDES})
        target_link_libraries(vrambridge-kv-cold-tier-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES})
        add_executable(vrambridge-kv-cold-gpu-test tests/test_kv_cold_gpu.cpp)
        target_include_directories(vrambridge-kv-cold-gpu-test PRIVATE
            ${VRAMBRIDGE_RENDER_INCLUDES} ${VRAMBRIDGE_CUDA12_RUNTIME_INCLUDE})
        target_link_libraries(vrambridge-kv-cold-gpu-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES}
            ${VRAMBRIDGE_CUDA12_RUNTIME})
    endif()
    if(VRAMBRIDGE_TEST_EXACT_PACKED_CPU)
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR
           NOT EXISTS "${VRAMBRIDGE_CPU_REFERENCE_LIBRARY}")
            message(FATAL_ERROR "Exact packed CPU tests need Linux x86_64 and an explicit installed CPU plugin")
        endif()
        add_library(vrambridge-exact-packed-cpu SHARED tools/trace/exact_packed_cpu.cpp)
        add_executable(vrambridge-exact-packed-cpu-test tests/test_exact_packed_cpu.cpp)
        foreach(TARGET IN ITEMS vrambridge-exact-packed-cpu vrambridge-exact-packed-cpu-test)
            target_include_directories(${TARGET} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace ${VRAMBRIDGE_RENDER_INCLUDES}
                ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/src ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/src/ggml-cpu)
            target_compile_options(${TARGET} PRIVATE
                -O3 -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mf16c -mfma -fno-fast-math -ffp-contract=off)
            target_link_libraries(${TARGET} PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES} ${CMAKE_DL_LIBS})
        endforeach()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/exact_packed_cpu.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/exact_packed_cpu.h)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/exact_packed_cpu.cpp VRAMBRIDGE_EXACT_PACKED_SOURCE_SHA256)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/exact_packed_cpu.h VRAMBRIDGE_EXACT_PACKED_HEADER_SHA256)
        target_compile_definitions(vrambridge-exact-packed-cpu PRIVATE
            VRAMBRIDGE_EXACT_PACKED_SOURCE_SHA256="${VRAMBRIDGE_EXACT_PACKED_SOURCE_SHA256}"
            VRAMBRIDGE_EXACT_PACKED_HEADER_SHA256="${VRAMBRIDGE_EXACT_PACKED_HEADER_SHA256}")
    endif()
    if(VRAMBRIDGE_TEST_ATTENTION_WINDOW)
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
            message(FATAL_ERROR "Attention-window interposition currently requires Linux")
        endif()
        add_library(vrambridge-attention-window SHARED tools/trace/attention_window.cpp)
        target_include_directories(vrambridge-attention-window PRIVATE ${VRAMBRIDGE_RENDER_INCLUDES})
        target_link_libraries(vrambridge-attention-window PRIVATE
            ${VRAMBRIDGE_TRACE_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/attention_window.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/attention_window.h)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/attention_window.cpp VRAMBRIDGE_ATTN_SOURCE_SHA256)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/attention_window.h VRAMBRIDGE_ATTN_HEADER_SHA256)
        target_compile_definitions(vrambridge-attention-window PRIVATE
            VRAMBRIDGE_ATTN_WINDOW_SOURCE_SHA256="${VRAMBRIDGE_ATTN_SOURCE_SHA256}"
            VRAMBRIDGE_ATTN_WINDOW_HEADER_SHA256="${VRAMBRIDGE_ATTN_HEADER_SHA256}")
        add_executable(vrambridge-attention-window-test tests/test_attention_window.cpp)
        target_include_directories(vrambridge-attention-window-test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace ${VRAMBRIDGE_RENDER_INCLUDES})
        target_link_libraries(vrambridge-attention-window-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES})
        add_executable(vrambridge-native-quantize
            ${VRAMBRIDGE_LLAMA_SOURCE}/tools/quantize/main.cpp
            ${VRAMBRIDGE_LLAMA_SOURCE}/tools/quantize/quantize.cpp)
        target_include_directories(vrambridge-native-quantize PRIVATE ${VRAMBRIDGE_RENDER_INCLUDES})
        target_link_libraries(vrambridge-native-quantize PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES} Threads::Threads)
    endif()
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
    if(VRAMBRIDGE_CPU_RESEARCH)
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            message(FATAL_ERROR "CPU research shims require Linux x86-64")
        endif()
        add_library(vrambridge-cpu-graph-profile SHARED tools/trace/cpu_graph_profile.cpp)
        target_include_directories(vrambridge-cpu-graph-profile PRIVATE
            ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include ${VRAMBRIDGE_LLAMA_SOURCE}/vendor)
        target_link_libraries(vrambridge-cpu-graph-profile PRIVATE ${VRAMBRIDGE_ggml_base_LIBRARY}
            Threads::Threads ${CMAKE_DL_LIBS})
        if(VRAMBRIDGE_CPU_PROFILE_NVTX)
            find_path(VRAMBRIDGE_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
                HINTS "$ENV{CUDA_PATH}/include" /usr/local/cuda/include REQUIRED)
            target_include_directories(vrambridge-cpu-graph-profile PRIVATE ${VRAMBRIDGE_NVTX_INCLUDE_DIR})
            target_compile_definitions(vrambridge-cpu-graph-profile PRIVATE VRAMBRIDGE_CPU_PROFILE_NVTX=1)
        endif()
        add_library(vrambridge-cpu-mxfp4-vnni SHARED tools/trace/cpu_mxfp4_vnni.cpp)
        target_compile_options(vrambridge-cpu-mxfp4-vnni PRIVATE
            -mavx2 -mfma -mf16c -mavx512vl -mavx512vnni)
        target_compile_definitions(vrambridge-hybrid-layer PRIVATE VRAMBRIDGE_CPU_CACHE_STAGING=1)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/cpu_cache_staging.h)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/cpu_cache_staging.h VRAMBRIDGE_CPU_CACHE_SOURCE_SHA256)
        target_compile_definitions(vrambridge-hybrid-layer PRIVATE
            VRAMBRIDGE_CPU_CACHE_SOURCE_SHA256="${VRAMBRIDGE_CPU_CACHE_SOURCE_SHA256}")
        add_executable(vrambridge-cpu-profile-hint-test tests/test_cpu_profile_hint.cpp)
        target_include_directories(vrambridge-cpu-profile-hint-test PRIVATE ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include)
        target_link_libraries(vrambridge-cpu-profile-hint-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES})
        target_compile_definitions(vrambridge-cpu-profile-hint-test PRIVATE
            VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}")
        add_library(vrambridge-cpu-live-warmer SHARED tools/trace/cpu_live_warmer.cpp)
        target_include_directories(vrambridge-cpu-live-warmer PRIVATE
            ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include ${VRAMBRIDGE_LLAMA_SOURCE}/vendor)
        target_link_libraries(vrambridge-cpu-live-warmer PRIVATE
            ${VRAMBRIDGE_ggml_base_LIBRARY} Threads::Threads ${CMAKE_DL_LIBS})
        if(VRAMBRIDGE_CPU_PROFILE_NVTX)
            target_include_directories(vrambridge-cpu-live-warmer PRIVATE ${VRAMBRIDGE_NVTX_INCLUDE_DIR})
            target_compile_definitions(vrambridge-cpu-live-warmer PRIVATE VRAMBRIDGE_CPU_PROFILE_NVTX=1)
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/cpu_live_warmer.cpp)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/cpu_live_warmer.cpp VRAMBRIDGE_LIVE_WARMER_SOURCE_SHA256)
        target_compile_definitions(vrambridge-cpu-live-warmer PRIVATE
            VRAMBRIDGE_LIVE_WARMER_SOURCE_SHA256="${VRAMBRIDGE_LIVE_WARMER_SOURCE_SHA256}")
        add_executable(vrambridge-live-warmer-test tests/test_live_warmer.cpp)
        target_include_directories(vrambridge-live-warmer-test PRIVATE
            ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include ${VRAMBRIDGE_LLAMA_SOURCE}/vendor)
        target_link_libraries(vrambridge-live-warmer-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES})
        target_compile_definitions(vrambridge-live-warmer-test PRIVATE
            VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}")
        add_library(vrambridge-cpu-control-threads SHARED tools/trace/cpu_control_threads.cpp)
        target_include_directories(vrambridge-cpu-control-threads PRIVATE
            ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include ${VRAMBRIDGE_LLAMA_SOURCE}/vendor)
        target_link_libraries(vrambridge-cpu-control-threads PRIVATE
            ${VRAMBRIDGE_ggml_base_LIBRARY} Threads::Threads ${CMAKE_DL_LIBS})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/cpu_control_threads.cpp)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/cpu_control_threads.cpp VRAMBRIDGE_CONTROL_THREADS_SOURCE_SHA256)
        target_compile_definitions(vrambridge-cpu-control-threads PRIVATE
            VRAMBRIDGE_CONTROL_THREADS_SOURCE_SHA256="${VRAMBRIDGE_CONTROL_THREADS_SOURCE_SHA256}")
        add_executable(vrambridge-cpu-control-threads-test tests/test_cpu_control_threads.cpp)
        target_include_directories(vrambridge-cpu-control-threads-test PRIVATE ${VRAMBRIDGE_LLAMA_SOURCE}/ggml/include)
        target_link_libraries(vrambridge-cpu-control-threads-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES})
        target_compile_definitions(vrambridge-cpu-control-threads-test PRIVATE
            VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}")
    endif()
    if(VRAMBRIDGE_TEST_COMPATIBLE_DOWN)
        if(NOT VRAMBRIDGE_ENABLE_CUDA OR NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR "Compatible-down research requires an enabled CUDA compiler")
        endif()
        target_sources(vrambridge-hybrid-layer PRIVATE tools/trace/compatible_down.cu)
        set_target_properties(vrambridge-hybrid-layer PROPERTIES
            CUDA_ARCHITECTURES native CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
        target_compile_options(vrambridge-hybrid-layer PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--fmad=false>")
        target_link_libraries(vrambridge-hybrid-layer PRIVATE CUDA::cudart)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/compatible_down.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/compatible_down.h)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/compatible_down.cu VRAMBRIDGE_COMPATIBLE_CUDA_SHA256)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/compatible_down.h VRAMBRIDGE_COMPATIBLE_HEADER_SHA256)
        target_compile_definitions(vrambridge-hybrid-layer PRIVATE VRAMBRIDGE_COMPATIBLE_DOWN=1
            VRAMBRIDGE_COMPATIBLE_CUDA_SHA256="${VRAMBRIDGE_COMPATIBLE_CUDA_SHA256}"
            VRAMBRIDGE_COMPATIBLE_HEADER_SHA256="${VRAMBRIDGE_COMPATIBLE_HEADER_SHA256}")
    endif()
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
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/logit_check.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/prefill_localization.h)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/logit_check.cpp VRAMBRIDGE_LOGIT_SOURCE_SHA256)
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tools/trace/prefill_localization.h VRAMBRIDGE_LOCALIZATION_SOURCE_SHA256)
        target_compile_definitions(vrambridge-logit-check PRIVATE
            VRAMBRIDGE_LLAMA_LIBRARY_DIR="${VRAMBRIDGE_LLAMA_LIBRARY_DIR}"
            VRAMBRIDGE_LOGIT_SOURCE_SHA256="${VRAMBRIDGE_LOGIT_SOURCE_SHA256}"
            VRAMBRIDGE_LOCALIZATION_SOURCE_SHA256="${VRAMBRIDGE_LOCALIZATION_SOURCE_SHA256}")
    endif()
    if(VRAMBRIDGE_TEST_PREFILL_PHASE)
        add_executable(vrambridge-prefill-phase-test tests/test_prefill_phase.cpp)
        get_target_property(VRAMBRIDGE_TRACE_INCLUDES vrambridge-trace INCLUDE_DIRECTORIES)
        target_include_directories(vrambridge-prefill-phase-test PRIVATE ${VRAMBRIDGE_TRACE_INCLUDES})
        target_link_libraries(vrambridge-prefill-phase-test PRIVATE ${VRAMBRIDGE_TRACE_LIBRARIES})
    endif()
endif()

if(VRAMBRIDGE_TEST_COMPATIBLE_DOWN AND NOT TARGET vrambridge-hybrid-layer)
    message(FATAL_ERROR "Compatible-down research requires pinned llama.cpp source and matching libraries")
endif()
if(VRAMBRIDGE_CPU_RESEARCH AND NOT TARGET vrambridge-cpu-graph-profile)
    message(FATAL_ERROR "CPU research requires pinned llama.cpp source and matching libraries")
endif()
if(VRAMBRIDGE_CPU_PROFILE_NVTX AND NOT VRAMBRIDGE_CPU_RESEARCH)
    message(FATAL_ERROR "CPU NVTX annotations require the opt-in CPU research profiler")
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
install(FILES tools/vrambridge/__init__.py tools/vrambridge/__main__.py
    tools/vrambridge/telemetry.py tools/vrambridge/updates.py DESTINATION share/vrambridge)
install(FILES tools/vrambridge/hybrid-profile.example.json DESTINATION share/vrambridge)
install(DIRECTORY tools/vrambridge/web DESTINATION share/vrambridge)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME vrambridge-prefill-seed
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prefill_seed.py)
    add_test(NAME vrambridge-cache-window
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cache_window.py)
    add_test(NAME vrambridge-live-warmer-study
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_live_warmer_study.py)
    add_test(NAME vrambridge-critical-path
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_critical_path.py)
    add_test(NAME vrambridge-handoff-study
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_handoff_study.py)
    add_test(NAME vrambridge-architecture-study
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_architecture_study.py)
    add_test(NAME vrambridge-engine-comparison
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_engine_comparison.py)
    if(TARGET vrambridge-render-corpus)
        add_test(NAME vrambridge-render-corpus-help COMMAND vrambridge-render-corpus --help)
        add_test(NAME vrambridge-render-corpus-inputs
            COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_render_corpus.py
                $<TARGET_FILE:vrambridge-render-corpus>)
    endif()
    if(TARGET vrambridge-attention-window-test)
        add_test(NAME vrambridge-attention-window-views COMMAND vrambridge-attention-window-test)
    endif()
    if(TARGET vrambridge-exact-packed-cpu-test)
        add_test(NAME vrambridge-exact-packed-cpu
            COMMAND vrambridge-exact-packed-cpu-test ${VRAMBRIDGE_CPU_REFERENCE_LIBRARY})
        set_tests_properties(vrambridge-exact-packed-cpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if(TARGET vrambridge-kv-cold-tier-test)
        add_test(NAME vrambridge-kv-cold-tier-roles COMMAND vrambridge-kv-cold-tier-test)
    endif()

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
    add_test(NAME vrambridge-io-study-inputs
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_io_study.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/vrambridge-io-study.py
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME vrambridge-model-staging
            COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_staging.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark/vrambridge-model-staging.py
        )
    endif()
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
            COMMAND ${NODE_EXECUTABLE} --test ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monitor.mjs
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_preferences.mjs)
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
        if(VRAMBRIDGE_TEST_COMPATIBLE_DOWN)
            add_test(NAME vrambridge-compatible-down-fixture COMMAND vrambridge-hybrid-layer
                --self-test --compatible-down --rows 7 --samples 3 --repetitions 3 --threads 2)
            add_test(NAME vrambridge-compatible-down-dp4a-fixture COMMAND vrambridge-hybrid-layer
                --self-test --compatible-down --compatible-dp4a --rows 7 --samples 3 --repetitions 3 --threads 2)
            add_test(NAME vrambridge-compatible-down-reports
                COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_compatible_down.py
                    $<TARGET_FILE:vrambridge-hybrid-layer>)
        endif()
        if(VRAMBRIDGE_CPU_RESEARCH)
            add_test(NAME vrambridge-cpu-control-threads
                COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_control_threads.py
                    $<TARGET_FILE:vrambridge-cpu-control-threads-test>
                    $<TARGET_FILE:vrambridge-cpu-control-threads>
                    ${VRAMBRIDGE_LLAMA_LIBRARY_DIR}/libggml-cpu-zen4.so)
            add_test(NAME vrambridge-live-warmer
                COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_live_warmer.py
                    $<TARGET_FILE:vrambridge-live-warmer-test>
                    $<TARGET_FILE:vrambridge-cpu-live-warmer>
                    ${VRAMBRIDGE_LLAMA_LIBRARY_DIR}/libggml-cpu-zen4.so)
            add_test(NAME vrambridge-cpu-research
                COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_research.py
                    $<TARGET_FILE:vrambridge-hybrid-layer>
                    $<TARGET_FILE:vrambridge-cpu-graph-profile>
                    $<TARGET_FILE:vrambridge-cpu-mxfp4-vnni>
                    ${VRAMBRIDGE_LLAMA_LIBRARY_DIR}/libggml-cpu-zen4.so
                    $<TARGET_FILE:vrambridge-cpu-profile-hint-test>)
            add_test(NAME vrambridge-cpu-cache-staging
                COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_cache_staging.py
                    $<TARGET_FILE:vrambridge-hybrid-layer>)
        endif()
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
    if(TARGET vrambridge-prefill-phase-test)
        add_test(NAME vrambridge-prefill-phase COMMAND vrambridge-prefill-phase-test)
    endif()
endif()