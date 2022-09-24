cmake_minimum_required(VERSION 3.10)
project(_lfmc_gym)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "$ENV{LOCAL_INSTALL}")

########################
### Helper Functions ###
########################
MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH (child ${children})
        IF (IS_DIRECTORY ${curdir}/${child})
            LIST(APPEND dirlist ${child})
        ENDIF ()
    ENDFOREACH ()
    SET(${result} ${dirlist})
ENDMACRO()

####################
### Dependencies ###
####################
set(Dependencies)

# PyBind
find_package(pybind11 REQUIRED)

# Eigen3
find_package(Eigen3 REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# RaiSim
find_package(raisim CONFIG REQUIRED)

# Dependencies
include_directories(dependencies/networks_minimal/include)
include_directories(dependencies/actuation_dynamics/include)

######################
### Build and Link ###
######################
set(RAISIM_GYM_ENV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gym_envs)
SUBDIRLIST(SUBDIRS ${RAISIM_GYM_ENV_DIR})

######## Env ##########
FOREACH (subdir ${SUBDIRS})
    pybind11_add_module(${subdir} raisim_gym_torch/env/raisim_gym.cpp raisim_gym_torch/env/Yaml.cpp)
    target_link_libraries(${subdir} PRIVATE raisim::raisim ${Dependencies})
    target_include_directories(${subdir} PUBLIC ${EIGEN3_INCLUDE_DIRS} ${RAISIM_GYM_ENV_DIR}/${subdir} ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_torch/env)
    target_compile_options(${subdir} PRIVATE -fPIC -O3 -flto -mno-avx2)
    target_compile_definitions(${subdir} PRIVATE "-DRAISIM_GYM_TORCH_ENV_NAME=${subdir}")
    target_compile_definitions(${subdir} PRIVATE "$<$<CONFIG:Release>:EIGEN_NO_DEBUG>")
    target_compile_definitions(${subdir} PRIVATE EIGEN_DONT_PARALLELIZE)

    if (CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("[RAISIM_GYM] BUILDING THE DEBUG APP for ${subdir}")
        add_executable(${subdir}_debug_app raisim_gym_torch/env/debug_app.cpp raisim_gym_torch/env/Yaml.cpp)
        target_link_libraries(${subdir}_debug_app PRIVATE raisim::raisim)
        target_include_directories(${subdir}_debug_app PUBLIC gym_envs/${subdir} ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/raisim_gym_torch/env)
        target_compile_options(${subdir}_debug_app PRIVATE -fPIC -g -O0 -mno-avx2)
    endif ()
ENDFOREACH ()
