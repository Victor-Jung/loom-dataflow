#include "loom_tune.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace loom {
namespace tune {

namespace {

class MappingProgramTunePass
    : public PassWrapper<MappingProgramTunePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MappingProgramTunePass)

  MappingProgramTunePass() = default;
  MappingProgramTunePass(const MappingProgramTunePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "loom-tune-mapping-program";
  }
  StringRef getDescription() const override {
    return "Run a search policy over the loop tree of each mapping program "
           "(p00 function) and replace the function by the chosen tree";
  }

  Option<std::string> policy{*this, "policy",
                             llvm::cl::desc("Search policy name"),
                             llvm::cl::init("identity")};
  Option<std::string> options{
      *this, "options",
      llvm::cl::desc("Policy options (for 'fixed': the schedule string)"),
      llvm::cl::init("")};
  Option<bool> dumpTree{*this, "dump-tree",
                        llvm::cl::desc("Print the loop tree before and after"),
                        llvm::cl::init(false)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::unique_ptr<Policy> pol = createPolicy(policy);
    if (!pol) {
      module.emitError("unknown tuning policy '")
          << policy << "'; known: " << llvm::join(policyNames(), ", ");
      return signalPassFailure();
    }

    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp f) {
      if (!f.isExternal())
        funcs.push_back(f);
    });

    Tuner tuner;
    Budget budget;
    for (func::FuncOp f : funcs) {
      auto tree = LoopTree::build(f);
      if (failed(tree))
        return signalPassFailure();
      if (dumpTree) {
        llvm::errs() << "loop tree (before):\n";
        tree->print(llvm::errs());
      }
      auto result = pol->run(*tree, tuner, budget, options);
      if (failed(result))
        return signalPassFailure();
      if (dumpTree) {
        llvm::errs() << "loop tree (after):\n";
        result->print(llvm::errs());
      }
      if (failed(result->emitInPlace())) {
        f.emitError("emitting the tuned loop tree failed verification");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createMappingProgramTunePass(StringRef policy,
                                                   StringRef options,
                                                   bool dumpTree) {
  auto pass = std::make_unique<MappingProgramTunePass>();
  pass->policy = policy.str();
  pass->options = options.str();
  pass->dumpTree = dumpTree;
  return pass;
}

void registerMappingProgramTunePass() {
  PassRegistration<MappingProgramTunePass>();
}

} // namespace tune
} // namespace loom
