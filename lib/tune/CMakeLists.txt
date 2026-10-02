# Mapping-program tuning: loop tree, moves, search policies and the
# loom-tune-mapping-program pass.

add_library(loomTune STATIC
  src/loop_tree.cpp
  src/moves.cpp
  src/policies.cpp
  src/pass.cpp
)

target_include_directories(loomTune PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(loomTune PUBLIC
  LoomPassesCommon
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  LLVMSupport
)

target_compile_options(loomTune PRIVATE -fno-rtti)
