cmake_minimum_required(VERSION 3.29)
project(yolov8app)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/pytorch/2.2.0_8/share/cmake/Torch/")
find_package(Torch REQUIRED)

message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

find_package(OpenCV REQUIRED)
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")

add_executable(${PROJECT_NAME}
    main2.cpp
)

target_link_libraries(${PROJECT_NAME}
    "${TORCH_LIBRARIES}"
    "${OpenCV_LIBRARIES}"
)

target_compile_features(${PROJECT_NAME}
    PRIVATE
    cxx_std_20
)
