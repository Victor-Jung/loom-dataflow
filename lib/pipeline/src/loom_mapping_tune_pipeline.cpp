#include "loom_mapping_tune_pipeline.h"
#include "loom_tune.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "ADL/IR/ADLDialect.h"
#include "LoomDialect.h.inc"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace loom {
namespace pipeline {

std::tuple<std::string, std::string>
runMappingTunePipeline(const std::string &input_mlir_text,
                       const std::string &policy,
                       const std::string &options) {
  DialectRegistry registry;
  registry.insert<BuiltinDialect, func::FuncDialect, affine::AffineDialect,
                  memref::MemRefDialect, arith::ArithDialect,
                  tensor::TensorDialect, linalg::LinalgDialect,
                  scf::SCFDialect, bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, math::MathDialect,
                  mlir::adl::ADLDialect, loom::LoomDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto inputBuf = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(input_mlir_text), "input_mlir");
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(inputBuf), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sm, &context);
  if (!module)
    return {"Failed to parse input MLIR text", ""};

  std::string diagnostics;
  llvm::raw_string_ostream diagStream(diagnostics);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    diagStream << diag.str() << "\n";
    return success();
  });

  PassManager pm(&context);
  pm.addPass(loom::tune::createMappingProgramTunePass(policy, options));
  if (failed(pm.run(*module)))
    return {"Mapping-program tuning failed: " + diagnostics, ""};

  std::string output;
  llvm::raw_string_ostream os(output);
  OpPrintingFlags flags;
  flags.useLocalScope();
  module->print(os, flags);
  os << "\n";
  return {"", output};
}

} // namespace pipeline
} // namespace loom
