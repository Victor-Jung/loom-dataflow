#include "loom_tune.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace loom {
namespace tune {

std::string Move::str(const LoopTree &t) const {
  switch (kind) {
  case Kind::Interchange:
    return "interchange(" + t.nodes[a].name + "," + t.nodes[b].name + ")";
  }
  return "?";
}

namespace {

/// True when `v` is defined inside the region of `op`.
bool definedInside(Value v, Operation *op) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return op->isAncestor(arg.getOwner()->getParentOp()) ||
           arg.getOwner()->getParentOp() == op;
  return op->isAncestor(v.getDefiningOp());
}

LogicalResult checkInterchange(const LoopTree &t, unsigned a, unsigned b,
                               std::string *why) {
  auto fail = [&](StringRef msg) {
    if (why)
      *why = msg.str();
    return failure();
  };
  const Node &outer = t.nodes[a];
  const Node &inner = t.nodes[b];
  if (outer.kind != Node::Kind::Temporal || inner.kind != Node::Kind::Temporal)
    return fail("both loops must be scf.for");
  if (inner.parent != (int)a)
    return fail("inner loop is not directly nested in outer loop");
  auto outerFor = cast<scf::ForOp>(outer.op);
  auto innerFor = cast<scf::ForOp>(inner.op);
  if (outerFor.getNumRegionIterArgs() || innerFor.getNumRegionIterArgs())
    return fail("loops with iter_args cannot be interchanged");

  // Every other child of the outer loop is hoisted above it: it must be pure
  // and independent of the outer induction variable.
  llvm::DenseSet<Value> ivDependent{outerFor.getInductionVar()};
  for (unsigned c : outer.children) {
    if (c == b)
      continue;
    Operation *op = t.nodes[c].op;
    if (t.nodes[c].isLoop())
      return fail("outer loop body contains another loop");
    if (!isMemoryEffectFree(op))
      return fail("outer loop body contains a statement with memory effects: " +
                  op->getName().getStringRef().str());
    for (Value operand : op->getOperands())
      if (ivDependent.contains(operand))
        return fail("a statement in the outer loop body depends on its "
                    "induction variable");
  }
  for (Value bound :
       {innerFor.getLowerBound(), innerFor.getUpperBound(), innerFor.getStep()})
    if (bound == outerFor.getInductionVar())
      return fail("inner loop bounds depend on the outer induction variable");
  return success();
}

} // namespace

LogicalResult Tuner::isLegal(const LoopTree &t, const Move &m,
                             std::string *why) const {
  switch (m.kind) {
  case Move::Kind::Interchange:
    return checkInterchange(t, m.a, m.b, why);
  }
  return failure();
}

SmallVector<Move> Tuner::legalMoves(const LoopTree &t) const {
  SmallVector<Move> out;
  for (unsigned li : t.loops())
    for (unsigned c : t.nodes[li].children)
      if (t.nodes[c].isLoop()) {
        Move m{Move::Kind::Interchange, li, c};
        if (succeeded(isLegal(t, m)))
          out.push_back(m);
      }
  return out;
}

FailureOr<LoopTree> Tuner::apply(const LoopTree &t, const Move &m) const {
  if (failed(isLegal(t, m)))
    return failure();
  LoopTree out = t;
  switch (m.kind) {
  case Move::Kind::Interchange: {
    Node &outer = out.nodes[m.a];
    Node &inner = out.nodes[m.b];
    Node &parent = out.nodes[outer.parent];

    // Parent: [.., outer, ..] -> [.., hoisted siblings.., inner, ..]
    SmallVector<unsigned, 4> replacement;
    for (unsigned c : outer.children)
      if (c != m.b)
        replacement.push_back(c);
    replacement.push_back(m.b);
    auto pos = llvm::find(parent.children, m.a);
    unsigned at = pos - parent.children.begin();
    parent.children.erase(pos);
    parent.children.insert(parent.children.begin() + at, replacement.begin(),
                           replacement.end());
    for (unsigned c : replacement)
      out.nodes[c].parent = outer.parent;

    // inner takes outer's place; outer takes inner's body.
    outer.children = inner.children;
    for (unsigned c : outer.children)
      out.nodes[c].parent = m.a;
    inner.children.assign({m.a});
    outer.parent = m.b;
    break;
  }
  }
  return out;
}

} // namespace tune
} // namespace loom
