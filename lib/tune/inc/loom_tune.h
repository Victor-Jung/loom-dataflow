/// Mapping-program tuning: a loop tree over a p00 function, moves on that
/// tree, and pluggable search policies.
///
/// A mapping program is a p00-level function: `affine.parallel` for the
/// spatial loops, `scf.for` for the temporal ones, `linalg` statements and
/// `memref.subview` accesses with symbolic (`loom.sym`) tile sizes. The tree
/// holds the real operations of one such function; a move edits the tree, and
/// `emit` materializes the edited tree as a fresh function next to the
/// original. The original IR is never rewritten in place.

#ifndef LOOM_TUNE_H
#define LOOM_TUNE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace tune {

/// One node of the loop tree. Nodes are stored in `LoopTree::nodes` and refer
/// to each other by index so that a tree is a plain value type.
struct Node {
  enum class Kind {
    Func,     ///< the function body (root)
    Spatial,  ///< affine.parallel
    Temporal, ///< scf.for
    Stmt      ///< any other top-level op of a loop body
  };
  Kind kind;
  mlir::Operation *op;
  int parent = -1;
  llvm::SmallVector<unsigned, 4> children; ///< loops and Func only, in program order
  std::string name;                        ///< loops only: "m", "n", "b" or "L<i>"
  unsigned loopId = 0;                     ///< loops only: pre-order index

  bool isLoop() const {
    return kind == Kind::Spatial || kind == Kind::Temporal;
  }
};

/// Loop tree of one function. Copying a tree copies the structure only; all
/// nodes keep pointing at the same operations.
struct LoopTree {
  mlir::func::FuncOp func;
  std::vector<Node> nodes;
  unsigned root = 0;

  /// Build the tree of `func`. Fails on loops with results other than scf.for
  /// iter_args (e.g. affine.parallel reductions).
  static mlir::FailureOr<LoopTree> build(mlir::func::FuncOp func);

  /// Node indices of all loops in pre-order (index i has loopId i).
  llvm::SmallVector<unsigned> loops() const;
  /// Loop node index by name or by "L<id>"; -1 when absent.
  int findLoop(llvm::StringRef nameOrId) const;

  /// Print the tree, one node per line, loops labelled with their id and name.
  void print(llvm::raw_ostream &os) const;

  /// Materialize the tree as a new function named `newName`, inserted right
  /// after `func`. Loops are recreated, statements cloned, values remapped.
  mlir::FailureOr<mlir::func::FuncOp> emit(llvm::StringRef newName) const;

  /// Emit the tree and replace `func` with the result, keeping its name.
  mlir::LogicalResult emitInPlace();
};

/// A primitive edit of the loop tree.
struct Move {
  enum class Kind { Interchange };
  Kind kind;
  unsigned a; ///< node index of the outer loop
  unsigned b; ///< node index of the inner loop

  std::string str(const LoopTree &t) const;
};

struct Budget {
  unsigned maxStates = 1000;
};

/// Legality and application of moves. Framework-owned: a policy reaches new
/// trees only through `apply`, so it cannot produce an illegal tree.
class Tuner {
public:
  llvm::SmallVector<Move> legalMoves(const LoopTree &t) const;
  mlir::LogicalResult isLegal(const LoopTree &t, const Move &m,
                              std::string *why = nullptr) const;
  mlir::FailureOr<LoopTree> apply(const LoopTree &t, const Move &m) const;
};

/// A search policy: decides which trees to spend the budget on.
class Policy {
public:
  virtual ~Policy() = default;
  virtual llvm::StringRef name() const = 0;
  /// `options` is the policy-specific option string (e.g. a schedule).
  virtual mlir::FailureOr<LoopTree> run(const LoopTree &start,
                                        const Tuner &tuner,
                                        const Budget &budget,
                                        llvm::StringRef options) = 0;
};

/// Instantiate a registered policy by name, or nullptr if unknown.
std::unique_ptr<Policy> createPolicy(llvm::StringRef name);
/// Names of all registered policies.
llvm::SmallVector<llvm::StringRef> policyNames();

/// Parse a schedule string ("interchange(m,n); interchange(L1,L2)") against
/// `t`. Loop references are names or "L<id>".
mlir::FailureOr<llvm::SmallVector<Move>> parseSchedule(llvm::StringRef text,
                                                       const LoopTree &t);

/// Run `policy` on every function of a module and replace each function by
/// the emitted result.
std::unique_ptr<mlir::Pass>
createMappingProgramTunePass(llvm::StringRef policy = "identity",
                             llvm::StringRef options = "",
                             bool dumpTree = false);
void registerMappingProgramTunePass();

} // namespace tune
} // namespace loom

#endif // LOOM_TUNE_H
