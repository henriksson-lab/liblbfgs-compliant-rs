cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(liblbfgs)
enable_language(C)

set(LBFGS_USE_SSE ON)
set(LBFGS_lib_TARGET_NAME lbfgs)
set(LBFGS_NO_ALIASES ON)
set(LBFGS_INSTALL_STATIC_LIBS ON)
set(LBFGS_INSTALL_HEADERS ON)
set(LBFGS_INSTALL_CONFIG OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/liblbfgs)
set_target_properties(
	lbfgs
	PROPERTIES
	DEBUG_POSTFIX ""
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${PROJECT_SOURCE_DIR}/Config.cmake.in"
	"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	INSTALL_DESTINATION "lib/liblbfgs/cmake"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION "lib/liblbfgs/cmake" COMPONENT Development)
