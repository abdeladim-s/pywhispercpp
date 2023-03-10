cmake_minimum_required(VERSION 3.4...3.18)
project(pywhispercpp)

add_subdirectory(pybind11)
add_subdirectory(whisper.cpp)

# file (GLOB CPP_FILES "whisper.cpp/*.cpp" "whisper.cpp/examples/*.cpp")
file (GLOB CPP_FILES "whisper.cpp/*.cpp")
file (GLOB C_FILES "whisper.cpp/*.c")


pybind11_add_module(_pywhispercpp
	${CPP_FILES}
	${C_FILES}
	src/main.cpp
)

# pybind11_add_module(pywhispercpp  src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_pywhispercpp
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
# ---------------------------------------------------------------------------------------------


option(WHISPER_SUPPORT_SDL2            "whisper: support for libSDL2" OFF)

if (APPLE)
    option(WHISPER_NO_ACCELERATE       "whisper: disable Accelerate framework" OFF)
    option(WHISPER_NO_AVX              "whisper: disable AVX" OFF)
    option(WHISPER_NO_AVX2             "whisper: disable AVX2" OFF)
    option(WHISPER_NO_FMA              "whisper: disable FMA" OFF)
else()
    option(WHISPER_SUPPORT_OPENBLAS    "whisper: support for OpenBLAS" OFF)
endif()

option(WHISPER_PERF                    "whisper: enable perf timings" OFF)



# sanitizers

if (NOT MSVC)
    if (WHISPER_SANITIZE_THREAD)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fsanitize=thread")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    endif()

    if (WHISPER_SANITIZE_ADDRESS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    endif()

    if (WHISPER_SANITIZE_UNDEFINED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=undefined")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    endif()
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

# -------------------------------------------
# dependencies

find_package(Threads REQUIRED)


# Compilation flags, from whisper.cpp

# on APPLE - include Accelerate framework
if (APPLE AND NOT WHISPER_NO_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        set(WHISPER_EXTRA_LIBS  ${WHISPER_EXTRA_LIBS}  ${ACCELERATE_FRAMEWORK})
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_USE_ACCELERATE)
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (WHISPER_SUPPORT_OPENBLAS)
    find_library(OPENBLAS_LIB
        NAMES openblas libopenblas
        )
    if (OPENBLAS_LIB)
        message(STATUS "OpenBLAS found")

        set(WHISPER_EXTRA_LIBS  ${WHISPER_EXTRA_LIBS}  ${OPENBLAS_LIB})
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_USE_OPENBLAS)
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

# compiler flags

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif ()

if (WHISPER_ALL_WARNINGS)
    if (NOT MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wshadow                        \
            -Wcast-qual                     \
            -Wstrict-prototypes             \
            -Wpointer-arith                 \
            -Wno-unused-function            \
        ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wcast-qual                     \
        ")
    else()
        # todo : msvc
    endif()
endif()

if (NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror=vla")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno -ffinite-math-only -funsafe-math-optimizations")
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
else()
    message(STATUS "x86 detected")
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
    else()
        if (EMSCRIPTEN)
            set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -pthread")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
        else()
            if(NOT WHISPER_NO_AVX)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
            endif()
            if(NOT WHISPER_NO_AVX2)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
            endif()
            if(NOT WHISPER_NO_FMA)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
            endif()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
        endif()
    endif()
endif()

if (WHISPER_PERF)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_PERF)
endif()