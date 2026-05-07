# cmake/FetchDependency.cmake
#
# Function to fetch a dependency with local path priority.
#
# Usage:
#   auto_fetch_dependency(<name> <git_repo> <git_tag>)
#
# Behavior:
#   1. If third_party/<name>/CMakeLists.txt exists, use it directly.
#   2. Else if previously downloaded, reuse cached version.
#   3. Otherwise, download from Git repository using FetchContent.
#
# @param name Dependency name (e.g., "googletest", "abseil-cpp")
# @param git_repo Git repository URL
# @param git_tag Git tag or branch to checkout
#
function(auto_fetch_dependency NAME GIT_REPO GIT_TAG)
    set(LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${NAME}")
    set(CACHE_FLAG "${CMAKE_CURRENT_BINARY_DIR}/_${NAME}_fetched")

    if(EXISTS "${LOCAL_PATH}/CMakeLists.txt")
        message(STATUS "Using local ${NAME} from ${LOCAL_PATH}")
        add_subdirectory("${LOCAL_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
    elseif(EXISTS "${CACHE_FLAG}")
        message(STATUS "Using previously fetched ${NAME} from cache")
        add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/${NAME}" "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
    else()
        message(STATUS "Fetching ${NAME} from ${GIT_REPO} (tag: ${GIT_TAG})...")
        include(FetchContent)
        FetchContent_Declare(
            "${NAME}"
            GIT_REPOSITORY "${GIT_REPO}"
            GIT_TAG        "${GIT_TAG}"
        )
        FetchContent_MakeAvailable("${NAME}")
        file(WRITE "${CACHE_FLAG}" "fetched")
    endif()
endfunction()