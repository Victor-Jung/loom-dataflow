#include "loom_tune.h"

#include "affine_utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace loom {
namespace tune {

namespace {

bool isLoopOp(Operation *op) {
  return isa<affine::AffineParallelOp, scf::ForOp>(op);
}

/// Name a loop after the tile symbol its trip count is derived from
/// ("tile_m" -> "m"). Multi-dimensional affine.parallel joins the names.
std::string loopNameFromBounds(Operation *op) {
  auto symName = [](Value v) -> std::string {
    if (auto ref = loom_affine::traceToLoomSymRef(v)) {
      StringRef leaf = ref->getLeafReference().getValue();
      leaf.consume_front("tile_");
      return leaf.str();
    }
    return "";
  };
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return symName(forOp.getUpperBound());
  if (auto par = dyn_cast<affine::AffineParallelOp>(op)) {
    std::string joined;
    for (Value v : par.getUpperBoundsOperands()) {
      std::string n = symName(v);
      if (n.empty())
        return "";
      if (!joined.empty())
        joined += ",";
      joined += n;
    }
    return joined;
  }
  return "";
}

} // namespace

FailureOr<LoopTree> LoopTree::build(func::FuncOp func) {
  LoopTree t;
  t.func = func;
  t.nodes.push_back(Node{Node::Kind::Func, func.getOperation()});
  t.root = 0;

  // Iterative pre-order walk; children are pushed in program order.
  SmallVector<unsigned> work{t.root};
  while (!work.empty()) {
    unsigned idx = work.pop_back_val();
    Operation *op = t.nodes[idx].op;
    Region &region = op->getRegion(0);
    if (!region.hasOneBlock())
      return op->emitError("loop tree: expected a single-block region");
    if (auto par = dyn_cast<affine::AffineParallelOp>(op))
      if (par.getNumResults() != 0)
        return op->emitError("loop tree: affine.parallel with results is "
                             "not supported");

    SmallVector<unsigned> kids;
    for (Operation &child : region.front().without_terminator()) {
      Node n;
      n.op = &child;
      n.parent = idx;
      if (isa<affine::AffineParallelOp>(child))
        n.kind = Node::Kind::Spatial;
      else if (isa<scf::ForOp>(child))
        n.kind = Node::Kind::Temporal;
      else
        n.kind = Node::Kind::Stmt;
      t.nodes.push_back(n);
      unsigned childIdx = t.nodes.size() - 1;
      t.nodes[idx].children.push_back(childIdx);
      if (isLoopOp(&child))
        kids.push_back(childIdx);
    }
    for (unsigned k : llvm::reverse(kids))
      work.push_back(k);
  }

  // Loop ids and names.
  llvm::StringSet<> used;
  unsigned id = 0;
  for (unsigned li : t.loops()) {
    Node &n = t.nodes[li];
    n.loopId = id++;
    std::string name = loopNameFromBounds(n.op);
    if (name.empty() || used.contains(name))
      name = "L" + std::to_string(n.loopId);
    used.insert(name);
    n.name = name;
  }
  return t;
}

SmallVector<unsigned> LoopTree::loops() const {
  SmallVector<unsigned> out;
  SmallVector<unsigned> work{root};
  while (!work.empty()) {
    unsigned idx = work.pop_back_val();
    if (nodes[idx].isLoop())
      out.push_back(idx);
    for (unsigned c : llvm::reverse(nodes[idx].children))
      if (nodes[c].isLoop())
        work.push_back(c);
  }
  return out;
}

int LoopTree::findLoop(StringRef nameOrId) const {
  for (unsigned li : loops()) {
    const Node &n = nodes[li];
    if (n.name == nameOrId)
      return li;
    if (nameOrId.consume_front("L")) {
      unsigned id;
      if (!nameOrId.getAsInteger(10, id) && id == n.loopId)
        return li;
    }
  }
  return -1;
}

void LoopTree::print(raw_ostream &os) const {
  unsigned stmtId = 0;
  std::function<void(unsigned, unsigned)> rec = [&](unsigned idx,
                                                    unsigned depth) {
    const Node &n = nodes[idx];
    os.indent(2 * depth);
    switch (n.kind) {
    case Node::Kind::Func:
      os << "func @" << func::FuncOp(func).getName() << "\n";
      break;
    case Node::Kind::Spatial:
      os << "L" << n.loopId << " spatial  " << n.name << "  ("
         << n.op->getName() << ")\n";
      break;
    case Node::Kind::Temporal: {
      auto forOp = cast<scf::ForOp>(n.op);
      os << "L" << n.loopId << " temporal " << n.name << "  (scf.for";
      if (forOp.getNumRegionIterArgs())
        os << ", " << forOp.getNumRegionIterArgs() << " iter_args";
      os << ")\n";
      break;
    }
    case Node::Kind::Stmt:
      os << "S" << stmtId++ << " " << n.op->getName() << "\n";
      return;
    }
    for (unsigned c : n.children)
      rec(c, depth + 1);
  };
  rec(root, 0);
}

FailureOr<func::FuncOp> LoopTree::emit(StringRef newName) const {
  OpBuilder builder(func->getContext());
  builder.setInsertionPointAfter(func);
  IRMapping mapping;

  // Recreate a region-holding op: clone it without its region, rebuild the
  // single block with equivalent arguments, emit the children, then clone the
  // original terminator so its operands are remapped.
  std::function<Operation *(unsigned)> emitRegionOp = [&](unsigned idx) {
    const Node &n = nodes[idx];
    Operation *oldOp = n.op;
    Operation *newOp = builder.cloneWithoutRegions(*oldOp, mapping);
    Block &oldBlock = oldOp->getRegion(0).front();
    SmallVector<Type> argTypes(oldBlock.getArgumentTypes());
    SmallVector<Location> argLocs;
    for (BlockArgument arg : oldBlock.getArguments())
      argLocs.push_back(arg.getLoc());
    Block *newBlock =
        builder.createBlock(&newOp->getRegion(0), {}, argTypes, argLocs);
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock.getArguments(), newBlock->getArguments()))
      mapping.map(oldArg, newArg);

    builder.setInsertionPointToEnd(newBlock);
    for (unsigned c : n.children) {
      if (nodes[c].isLoop())
        emitRegionOp(c);
      else
        builder.clone(*nodes[c].op, mapping);
    }
    builder.clone(*oldBlock.getTerminator(), mapping);
    builder.setInsertionPointAfter(newOp);
    return newOp;
  };

  auto newFunc = cast<func::FuncOp>(emitRegionOp(root));
  newFunc.setName(newName);
  if (failed(verify(newFunc))) {
    newFunc.erase();
    return failure();
  }
  return newFunc;
}

LogicalResult LoopTree::emitInPlace() {
  std::string name = func.getName().str();
  auto newFunc = emit(name + "__tuned");
  if (failed(newFunc))
    return failure();
  func.erase();
  newFunc->setName(name);
  func = *newFunc;
  return success();
}

} // namespace tune
} // namespace loom
