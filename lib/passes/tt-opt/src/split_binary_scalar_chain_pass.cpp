#include "Passes.h"
#include "binary_scalar_chain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "LoomDialect.h.inc"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace {

bool isScalarOrRankOne(Type type) {
  if (type.isIntOrIndexOrFloat())
    return true;

  auto shapedType = dyn_cast<ShapedType>(type);
  return shapedType && shapedType.hasRank() && shapedType.getRank() <= 1;
}

bool tracesToScalarOrRankOneInput(Value value, linalg::GenericOp genericOp,
                                  SmallPtrSetImpl<Operation *> &visitedOps) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg) {
    if (blockArg.getOwner() != genericOp.getBody())
      return false;

    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber >= genericOp.getNumDpsInputs())
      return false;

    Value genericInput = genericOp.getDpsInputs()[argNumber];
    return isScalarOrRankOne(genericInput.getType());
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getBlock() != genericOp.getBody())
    return false;

  if (!visitedOps.insert(defOp).second)
    return false;

  for (Value operand : defOp->getOperands()) {
    if (tracesToScalarOrRankOneInput(operand, genericOp, visitedOps))
      return true;
  }
  return false;
}

bool binaryOpUsesScalarOrRankOneInput(Operation *op,
                                      linalg::GenericOp genericOp) {
  SmallPtrSet<Operation *, 8> visitedOps;
  for (Value operand : op->getOperands()) {
    if (tracesToScalarOrRankOneInput(operand, genericOp, visitedOps))
      return true;
  }
  return false;
}

bool matchHasUnsplittableScalarOrRankOneInput(
    const loom::utils::BinaryScalarChainMatch &match) {
  return binaryOpUsesScalarOrRankOneInput(match.intermediateOp,
                                          match.genericOp) ||
         binaryOpUsesScalarOrRankOneInput(match.splitOp, match.genericOp);
}

SmallVector<Value, 4> getInputsByIndex(linalg::GenericOp genericOp,
                                       ArrayRef<unsigned> indices) {
  SmallVector<Value, 4> inputs;
  ValueRange originalInputs = genericOp.getDpsInputs();
  for (unsigned index : indices)
    inputs.push_back(originalInputs[index]);
  return inputs;
}

SmallVector<AffineMap, 4> getInputMapsByIndex(linalg::GenericOp genericOp,
                                              ArrayRef<unsigned> indices) {
  SmallVector<AffineMap, 4> maps;
  SmallVector<AffineMap> originalMaps = genericOp.getIndexingMapsArray();
  for (unsigned index : indices)
    maps.push_back(originalMaps[index]);
  return maps;
}

void mapSelectedInputBlockArgs(IRMapping &mapping, linalg::GenericOp genericOp,
                               ArrayRef<unsigned> indices,
                               ValueRange newBlockArgs,
                               unsigned newBlockArgOffset) {
  Block *oldBody = genericOp.getBody();
  for (auto [newArgIndex, oldInputIndex] : llvm::enumerate(indices)) {
    mapping.map(oldBody->getArgument(oldInputIndex),
                newBlockArgs[newBlockArgOffset + newArgIndex]);
  }
}

Operation *clonePayloadOp(OpBuilder &builder, Operation *op,
                          IRMapping &mapping) {
  return builder.clone(*op, mapping);
}

void buildFirstBody(OpBuilder &builder, Location loc,
                    loom::utils::BinaryScalarChainMatch &match,
                    ValueRange blockArgs) {
  IRMapping mapping;
  linalg::GenericOp genericOp = match.genericOp;
  unsigned numInputs = genericOp.getNumDpsInputs();

  mapSelectedInputBlockArgs(mapping, genericOp, match.firstInputIndices,
                            blockArgs, 0);
  mapping.map(genericOp.getBody()->getArgument(numInputs),
              blockArgs[match.firstInputIndices.size()]);

  for (Operation *op : match.firstOps)
    clonePayloadOp(builder, op, mapping);

  Value yielded = mapping.lookup(match.intermediateOp->getResult(0));
  builder.create<linalg::YieldOp>(loc, yielded);
}

void buildSecondBody(OpBuilder &builder, Location loc,
                     loom::utils::BinaryScalarChainMatch &match,
                     ValueRange blockArgs) {
  IRMapping mapping;
  linalg::GenericOp genericOp = match.genericOp;
  unsigned numInputs = genericOp.getNumDpsInputs();

  mapping.map(match.intermediateOp->getResult(0), blockArgs[0]);
  mapSelectedInputBlockArgs(mapping, genericOp, match.secondInputIndices,
                            blockArgs, 1);
  mapping.map(genericOp.getBody()->getArgument(numInputs),
              blockArgs[1 + match.secondInputIndices.size()]);

  for (Operation *op : match.secondOps)
    clonePayloadOp(builder, op, mapping);

  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value yielded = mapping.lookupOrDefault(yieldOp.getOperand(0));
  builder.create<linalg::YieldOp>(loc, yielded);
}


/// Follow only aliasing edges (destination-passing inits and views) back to the
/// buffer a value ultimately lives in. Unlike the general
/// `traceToRootAllocOp`, this never crosses into an operand that merely feeds a
/// computation, so two values compare equal here exactly when they share
/// storage.
Value rootBufferOf(Value value) {
  SmallPtrSet<Operation *, 8> seen;
  while (value) {
    Operation *def = value.getDefiningOp();
    if (!def || !seen.insert(def).second)
      return value;
    if (isa<loom::AllocOp>(def))
      return value;
    if (auto take = dyn_cast<loom::SemaphoreTakeOp>(def)) {
      value = take->getOperand(0);
      continue;
    }
    if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
      value = view.getViewSource();
      continue;
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(def)) {
      auto result = dyn_cast<OpResult>(value);
      if (result && result.getResultNumber() < dps.getNumDpsInits()) {
        value = dps.getDpsInits()[result.getResultNumber()];
        continue;
      }
    }
    return value;
  }
  return value;
}

bool splitBinaryScalarChain(loom::utils::BinaryScalarChainMatch match,
                            RewriterBase &rewriter) {
  linalg::GenericOp genericOp = match.genericOp;
  Value output = genericOp.getDpsInits()[0];
  SmallVector<AffineMap> originalMaps = genericOp.getIndexingMapsArray();
  unsigned numInputs = genericOp.getNumDpsInputs();
  AffineMap outputMap = originalMaps[numInputs];
  auto iteratorTypes = genericOp.getIteratorTypesArray();

  SmallVector<Value, 4> firstInputs =
      getInputsByIndex(genericOp, match.firstInputIndices);
  SmallVector<AffineMap, 4> firstMaps =
      getInputMapsByIndex(genericOp, match.firstInputIndices);
  firstMaps.push_back(outputMap);

  SmallVector<Value, 4> secondExtraInputs =
      getInputsByIndex(genericOp, match.secondInputIndices);

  // The split parks the chain's intermediate in the original destination. That
  // is only safe when no other input of the second op lives in that same
  // buffer: otherwise the first op's write destroys a value the second op still
  // reads. This pass runs after bufferization, so nothing downstream will
  // notice, and once lowered the buffer also takes two cb_push_back against a
  // single cb_pop_front -- the packer then blocks forever in a cb_reserve_back
  // that can never be satisfied. Give the intermediate its own buffer instead.
  Value outputBuffer = rootBufferOf(output);
  bool destAliasesInput =
      outputBuffer && llvm::any_of(secondExtraInputs, [&](Value in) {
        return rootBufferOf(in) == outputBuffer;
      });

  Value dest = output;
  if (destAliasesInput) {
    auto allocOp = outputBuffer.getDefiningOp<loom::AllocOp>();
    auto memrefType = dyn_cast<MemRefType>(output.getType());
    if (!allocOp || !memrefType)
      return false; // cannot give it separate storage; leave the op fused
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(allocOp);
    auto scratch = loom::AllocOp::create(
        rewriter, allocOp.getLoc(), allocOp.getResult().getType(),
        allocOp.getSizes(), allocOp.getStaticSizesAttr(),
        allocOp.getAlignmentAttr(), allocOp.getBufferCountAttr(),
        allocOp.getMemoryAttr());
    dest = loom::SemaphoreTakeOp::create(rewriter, allocOp.getLoc(), memrefType,
                                         scratch.getResult());
  }

  // Everything downstream of the chain wants the chain's result, which now
  // lands in `dest`. Collect those uses while the original op still pins the
  // program order.
  SmallVector<OpOperand *, 4> usesToRedirect;
  if (destAliasesInput) {
    Block *block = genericOp->getBlock();
    for (OpOperand &use : output.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == genericOp || isa<loom::SemaphoreGiveOp>(owner))
        continue;
      Operation *ancestor = block->findAncestorOpInBlock(*owner);
      if (ancestor && genericOp->isBeforeInBlock(ancestor))
        usesToRedirect.push_back(&use);
    }
  }

  SmallVector<Value, 4> secondInputs;
  secondInputs.push_back(dest);
  llvm::append_range(secondInputs, secondExtraInputs);

  SmallVector<AffineMap, 4> secondMaps;
  secondMaps.push_back(outputMap);
  llvm::append_range(secondMaps,
                     getInputMapsByIndex(genericOp, match.secondInputIndices));
  secondMaps.push_back(outputMap);

  Location loc = genericOp.getLoc();
  rewriter.setInsertionPoint(genericOp);
  rewriter.create<linalg::GenericOp>(
      loc, firstInputs, ValueRange(dest), firstMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        buildFirstBody(nestedBuilder, nestedLoc, match, args);
      });

  rewriter.create<linalg::GenericOp>(
      loc, secondInputs, ValueRange(dest), secondMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        buildSecondBody(nestedBuilder, nestedLoc, match, args);
      });

  rewriter.eraseOp(genericOp);

  if (destAliasesInput) {
    for (OpOperand *use : usesToRedirect)
      use->set(dest);
    // Release the new buffer wherever the original destination is released.
    for (Operation *user : output.getUsers()) {
      if (auto give = dyn_cast<loom::SemaphoreGiveOp>(user)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(give);
        loom::SemaphoreGiveOp::create(rewriter, give.getLoc(), dest);
        break;
      }
    }
  }
  return true;
}

struct SplitBinaryScalarChainPass
    : public PassWrapper<SplitBinaryScalarChainPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitBinaryScalarChainPass)

  StringRef getArgument() const override {
    return "tt-split-binary-scalar-chain";
  }

  StringRef getDescription() const override {
    return "Split safe fused binary scalar chains in linalg.generic ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    constexpr unsigned kMaxDepth = 3;
    MLIRContext *context = &getContext();
    loom::utils::BinaryScalarChainAnalyzer analyzer;

    for (unsigned depth = 0; depth < kMaxDepth; ++depth) {
      SmallVector<linalg::GenericOp, 16> generics;
      getOperation().walk(
          [&](linalg::GenericOp op) { generics.push_back(op); });

      bool changed = false;
      IRRewriter rewriter(context);
      for (linalg::GenericOp genericOp : generics) {
        if (!genericOp->getParentOp())
          continue;

        std::optional<loom::utils::BinaryScalarChainMatch> match =
            analyzer.findFirstMatch(genericOp);
        if (!match)
          continue;
        if (matchHasUnsplittableScalarOrRankOneInput(*match))
          continue;

        changed |= splitBinaryScalarChain(*match, rewriter);
      }

      if (!changed)
        break;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
loom::passes::createSplitBinaryScalarChainPass() {
  return std::make_unique<SplitBinaryScalarChainPass>();
}
