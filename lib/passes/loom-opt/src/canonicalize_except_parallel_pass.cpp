#include "Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace loom {
namespace passes {

/// Canonicalize with every registered pattern except those of `scf.parallel`.
///
/// The upstream `scf.parallel` canonicalizer folds single-iteration
/// dimensions and rebuilds the op without its discardable attributes, which
/// erases the `loom.physical_dims` mapping of a spatial loop with one
/// iteration. After affine lowering the spatial loops must survive to the
/// backend untouched, so this replaces `-canonicalize` from that point on.
class CanonicalizeExceptParallelPass
    : public mlir::PassWrapper<CanonicalizeExceptParallelPass,
                               mlir::OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeExceptParallelPass)

  llvm::StringRef getArgument() const override {
    return "loom-canonicalize-except-parallel";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalize with all patterns except those of scf.parallel";
  }

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    for (mlir::RegisteredOperationName op : ctx->getRegisteredOperations())
      if (op.getStringRef() != mlir::scf::ParallelOp::getOperationName())
        op.getCanonicalizationPatterns(patterns, ctx);
    for (mlir::Dialect *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createCanonicalizeExceptParallelPass() {
  return std::make_unique<CanonicalizeExceptParallelPass>();
}

} // namespace passes
} // namespace loom
