# CMake toolchain file for Windows (MSVC) builds, set in .cargo/config.toml.
#
# llama.cpp requires CMake policies that pick the C runtime through
# CMAKE_MSVC_RUNTIME_LIBRARY (CMP0091) and ignore the /MT flag that
# llama-cpp-sys-2 passes for LLAMA_STATIC_CRT=1. Without this line llama.cpp
# is built against the DLL runtime and the link with the static rest fails.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
