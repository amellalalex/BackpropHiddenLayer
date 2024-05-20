# CMakeList.txt : CMake project for BackpropHiddenLayer, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (BackpropHiddenLayer "BackpropHiddenLayer.cpp" )

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET BackpropHiddenLayer PROPERTY CXX_STANDARD 20)
endif()

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(BackpropHiddenLayer PRIVATE Eigen3::Eigen)

# TODO: Add tests and install targets if needed.
