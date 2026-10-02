/**
 * @file utils.cpp
 * @brief Implementation of common utilities for function cloning.
 */

#include "utils.h"
#include "hardware_info.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cassert>
#include <set>
#include <string>

// Include the generated Loom dialect headers
#include "LoomDialect.h.inc"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "LoomInterfaces.h.inc"
#define GET_OP_CLASSES
#include "LoomOps.h.inc"

using namespace mlir;

namespace loom {
namespace utils {
namespace {

using SizeChoice = std::optional<int64_t>;

SmallVector<SizeChoice> getOccupancyChoices(const SpatialDimInfo &dim) {
  SmallVector<SizeChoice> choices;
  if (!dim.size || *dim.size < 2) {
    choices.push_back(dim.size);
    return choices;
  }

  for (int64_t size = 2; size <= *dim.size; size += 2)
    choices.push_back(size);
  return choices;
}

std::string buildOccupancyKey(ArrayRef<SizeChoice> sizes) {
  std::string key;
  llvm::raw_string_ostream os(key);
  bool first = true;
  for (SizeChoice size : sizes) {
    if (!first)
      os << ",";
    first = false;
    if (size)
      os << *size;
    else
      os << "?";
  }
  return key;
}

HardwareInfo withOccupancySizes(const HardwareInfo &base,
                                ArrayRef<SizeChoice> sizes) {
  HardwareInfo variant = base;
  for (auto [idx, size] : llvm::enumerate(sizes))
    variant.spatialDimInfoVec[idx].size = size;
  return variant;
}

} // namespace

ModuleOp getParentModule(func::FuncOp func) {
  Operation *parent = func->getParentOp();
  if (auto module = dyn_cast_or_null<ModuleOp>(parent)) {
    return module;
  }
  return nullptr;
}

func::FuncOp cloneFunc(OpBuilder &builder, func::FuncOp originalFunc,
                       llvm::StringRef newName, DictionaryAttr moduleAttrs,
                       std::function<LogicalResult(func::FuncOp)> modifier,
                       Operation *insertAfter) {
  if (insertAfter)
    builder.setInsertionPointAfter(insertAfter);

  ModuleOp wrapperModule = nullptr;
  OpBuilder effectiveBuilder = builder;
  if (moduleAttrs) {
    wrapperModule = ModuleOp::create(originalFunc.getLoc());
    wrapperModule->setAttrs(moduleAttrs);
    builder.insert(wrapperModule);
    effectiveBuilder = OpBuilder(wrapperModule.getBodyRegion());
  }

  IRMapping mapping;
  auto clonedFunc =
      cast<func::FuncOp>(effectiveBuilder.clone(*originalFunc, mapping));
  clonedFunc.setName(newName);

  if (modifier && failed(modifier(clonedFunc))) {
    if (wrapperModule)
      wrapperModule.erase();
    else
      clonedFunc.erase();
    return nullptr;
  }

  builder.setInsertionPointAfter(wrapperModule ? (Operation *)wrapperModule
                                               : (Operation *)clonedFunc);
  return clonedFunc;
}

llvm::SmallVector<func::FuncOp> collectFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> funcs;
  module.walk([&](func::FuncOp func) { funcs.push_back(func); });
  return funcs;
}

StringRef traceToSymbolicVar(Value val) {
  if (!val)
    return "";

  // Handle direct loom.sym
  if (auto getSym = val.getDefiningOp<loom::SymOp>()) {
    return getSym.getSymbolRef().getLeafReference().getValue();
  }

  // Handle arith.muli/addi/etc. if needed, but for now we follow the user's
  // sketch where block sizes are directly used from
  // loom.get_symbolic_block_size.

  return "";
}

SmallVector<HardwareInfo>
generateHardwareOccupancyVariants(const HardwareInfo &hardwareInfo) {
  SmallVector<HardwareInfo> variants;
  const unsigned dimCount =
      static_cast<unsigned>(hardwareInfo.spatialDimInfoVec.size());
  if (dimCount == 0) {
    variants.push_back(hardwareInfo);
    return variants;
  }

  SmallVector<SmallVector<SizeChoice>> choicesByDim;
  choicesByDim.reserve(dimCount);
  for (const SpatialDimInfo &dim : hardwareInfo.spatialDimInfoVec)
    choicesByDim.push_back(getOccupancyChoices(dim));

  std::set<std::string> seen;
  auto addVariant = [&](ArrayRef<SizeChoice> sizes) {
    std::string key = buildOccupancyKey(sizes);
    if (!seen.insert(key).second)
      return;
    variants.push_back(withOccupancySizes(hardwareInfo, sizes));
  };

  SmallVector<SizeChoice> current;
  current.reserve(dimCount);
  std::function<void(unsigned)> enumerate = [&](unsigned dimIdx) {
    if (dimIdx == dimCount) {
      addVariant(current);
      return;
    }
    for (SizeChoice choice : choicesByDim[dimIdx]) {
      current.push_back(choice);
      enumerate(dimIdx + 1);
      current.pop_back();
    }
  };
  enumerate(0);
  return variants;
}

namespace {

/// The nearest enclosing affine.parallel or non-loop-carried scf.for.
Operation *memoryBindingContainer(Operation *op) {
  Operation *p = op->getParentOp();
  while (p) {
    if (isa<affine::AffineParallelOp>(p))
      return p;
    if (auto forOp = dyn_cast<scf::ForOp>(p))
      if (forOp.getInitArgs().empty())
        return p;
    p = p->getParentOp();
  }
  return nullptr;
}

Type findL1AllocElementType(loom::AllocOp alloc) {
  for (auto user : alloc.getResult().getUsers()) {
    if (auto init = dyn_cast<loom::InitTensorOp>(user))
      return cast<RankedTensorType>(init.getResult().getType()).getElementType();
    if (auto copy = dyn_cast<loom::CopyToTensorOp>(user))
      return cast<RankedTensorType>(copy.getResult().getType()).getElementType();
  }
  return Type{};
}

/// Discriminates a dynamic alloc operand: either a plain symbolic variable
/// or a `(numerator ceildiv symbolic)` form (from `arith.ceildivsi` or
/// `affine.apply` with a CeilDiv map).
struct DynDim {
  StringRef sym;                                       // plain symbolic
  std::optional<std::pair<int64_t, StringRef>> ceildiv; // (numerator, denomSym)
};

/// Try to recognize `(constant numerator ceildiv symbolic-denominator)` from
/// either `arith.ceildivsi` or an `affine.apply` whose map is a CeilDiv with a
/// constant LHS and a single symbol RHS. Returns the (numerator, denomSym)
/// pair or nullopt.
std::optional<std::pair<int64_t, StringRef>>
matchConstCeildivSym(Value val) {
  if (auto ceildiv = val.getDefiningOp<arith::CeilDivSIOp>()) {
    int64_t numerator = -1;
    if (auto c = ceildiv.getLhs().getDefiningOp<arith::ConstantIndexOp>())
      numerator = c.value();
    else if (auto c = ceildiv.getLhs().getDefiningOp<arith::ConstantIntOp>())
      numerator = c.value();
    if (numerator <= 0)
      return std::nullopt;
    StringRef denom = traceToSymbolicVar(ceildiv.getRhs());
    if (denom.empty())
      return std::nullopt;
    return std::make_pair(numerator, denom);
  }

  if (auto apply = val.getDefiningOp<affine::AffineApplyOp>()) {
    auto map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return std::nullopt;
    auto expr = map.getResult(0);
    if (expr.getKind() != AffineExprKind::CeilDiv)
      return std::nullopt;
    auto binary = cast<AffineBinaryOpExpr>(expr);
    auto lhs = binary.getLHS();
    auto rhs = binary.getRHS();
    if (lhs.getKind() != AffineExprKind::Constant ||
        rhs.getKind() != AffineExprKind::SymbolId)
      return std::nullopt;
    int64_t numerator = cast<AffineConstantExpr>(lhs).getValue();
    unsigned symIdx = cast<AffineSymbolExpr>(rhs).getPosition();
    if (symIdx >= apply.getNumOperands())
      return std::nullopt;
    StringRef denom = traceToSymbolicVar(apply.getOperand(symIdx));
    if (denom.empty())
      return std::nullopt;
    return std::make_pair(numerator, denom);
  }

  return std::nullopt;
}

DynDim classifyDynamicAllocOperand(Value val) {
  if (auto cd = matchConstCeildivSym(val))
    return DynDim{StringRef{}, cd};
  return DynDim{traceToSymbolicVar(val), std::nullopt};
}

/// Cancel `block_K * (K_total ceildiv block_K) -> K_total`: when a ceildiv's
/// denominator symbol appears among `syms`, drop it from `syms` and fold the
/// numerator into `elemSize` (a constant total-size multiplier).
void cancelHoistedDivPairs(
    SmallVectorImpl<StringRef> &syms,
    ArrayRef<std::pair<int64_t, StringRef>> ceildivs,
    int64_t &elemSize) {
  for (const auto &[numerator, denom] : ceildivs) {
    auto it = llvm::find(syms, denom);
    if (it != syms.end()) {
      syms.erase(it);
      elemSize *= numerator;
    }
  }
}

} // namespace

Operation *
getMemoryBindingScope(llvm::ArrayRef<Operation *> definingOps) {
  Operation *scope = nullptr;
  for (Operation *def : definingOps) {
    Operation *container = memoryBindingContainer(def);
    if (!container)
      return nullptr;
    if (!scope) {
      scope = container;
      continue;
    }
    while (scope != container && !scope->isAncestor(container)) {
      scope = memoryBindingContainer(scope);
      if (!scope)
        return nullptr;
    }
  }
  return scope;
}

llvm::SmallVector<AllocInfo> collectL1AllocInfos(func::FuncOp func) {
  llvm::SmallVector<AllocInfo> allocInfos;

  func.walk([&](loom::AllocOp alloc) {
    if (alloc.getMemory().getLeafReference() != "L1")
      return;

    Type elementType = findL1AllocElementType(alloc);
    if (!elementType)
      return;

    int64_t baseElemSize = elementType.getIntOrFloatBitWidth() / 8;
    AllocInfo info;
    info.elemSize = baseElemSize * alloc.getBufferCount();

    SmallVector<StringRef> syms;
    SmallVector<std::pair<int64_t, StringRef>> ceildivs;
    for (Value val : alloc.getSizes()) {
      DynDim d = classifyDynamicAllocOperand(val);
      if (d.ceildiv)
        ceildivs.push_back(*d.ceildiv);
      else if (!d.sym.empty())
        syms.push_back(d.sym);
    }

    cancelHoistedDivPairs(syms, ceildivs, info.elemSize);
    info.dims = syms;

    // Record only if it has symbolic dims or total size exceeds a single elem
    // (fixed-size multi-buffer or hoisted buffer).
    if (!info.dims.empty() || info.elemSize > baseElemSize)
      allocInfos.push_back(std::move(info));
  });

  return allocInfos;
}

} // namespace utils
} // namespace loom
