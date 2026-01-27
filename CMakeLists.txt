cmake_minimum_required(VERSION 3.15)
project(pyquantization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

file(GLOB SOURCES
    src/*.cpp
    src/refine/*.cpp
)

# Exclude main.cpp if it somehow got copied
list(FILTER SOURCES EXCLUDE REGEX "main\\.cpp$")

pybind11_add_module(pyquantization ${SOURCES} bindings/pyquantization.cpp)

target_include_directories(pyquantization PRIVATE src)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pyquantization PRIVATE -Wall -Wextra -fdiagnostics-color)
    set_source_files_properties(src/refine/cgls.cpp PROPERTIES COMPILE_FLAGS -march=native)
endif()

if(MSVC)
    target_compile_options(pyquantization PRIVATE /W2 /bigobj /wd4267 /wd4244)
endif()

install(TARGETS pyquantization LIBRARY DESTINATION .)
