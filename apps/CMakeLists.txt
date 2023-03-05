function(build_metal_compute_shader output_file entry_point)
  set(BUILD_KERNEL_CMD   "xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${entry_point} -o ${CMAKE_BINARY_DIR}/kernel")
  set(BUNDLE_LIBRARY_CMD "xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/kernel                 -o ${output_file}")
  add_custom_command(
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${entry_point} -o ${CMAKE_BINARY_DIR}/${entry_point}_kernel
    COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/${entry_point}_kernel  -o ${output_file}
    OUTPUT "${output_file}"
    DEPENDS ${entry_point}
    VERBATIM)
endfunction()

function(create_binary target_name entry_point kernel_entry_point)
  add_executable(${target_name} ${entry_point})
  target_include_directories(
    ${target_name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src)
  build_metal_compute_shader(
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target_name}_library"
    ${kernel_entry_point})
  add_custom_target(
    "${target_name}_compute_shader"
    DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target_name}_library")
  add_dependencies(${target_name} "${target_name}_compute_shader")
  target_link_libraries(${target_name} ${METAL} ${FOUNDATION} ${METAL_KIT})
endfunction()

create_binary(cachesize cachesize.cpp cachesize.metal)
create_binary(cacheline cacheline.cpp cacheline.metal)
create_binary(cacheassociativity cacheassociativity.cpp cacheassociativity.metal)
create_binary(localmem localmem.cpp localmem.metal)