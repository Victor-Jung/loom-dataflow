#include "loom_tune.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace loom {
namespace tune {

FailureOr<SmallVector<Move>> parseSchedule(StringRef text, const LoopTree &t) {
  SmallVector<Move> moves;
  SmallVector<StringRef> items;
  text.split(items, ';', -1, /*KeepEmpty=*/false);
  for (StringRef item : items) {
    item = item.trim();
    if (item.empty())
      continue;
    size_t open = item.find('(');
    if (open == StringRef::npos || !item.ends_with(")"))
      return func::FuncOp(t.func).emitError("schedule: expected name(args): '") << item
             << "'";
    StringRef name = item.take_front(open).trim();
    SmallVector<StringRef> args;
    item.slice(open + 1, item.size() - 1).split(args, ',');
    for (StringRef &a : args)
      a = a.trim();

    if (name == "interchange") {
      if (args.size() != 2)
        return func::FuncOp(t.func).emitError("schedule: interchange takes 2 loops");
      int a = t.findLoop(args[0]), b = t.findLoop(args[1]);
      if (a < 0 || b < 0)
        return func::FuncOp(t.func).emitError("schedule: unknown loop in '") << item << "'";
      moves.push_back(Move{Move::Kind::Interchange, (unsigned)a, (unsigned)b});
    } else {
      return func::FuncOp(t.func).emitError("schedule: unknown move '") << name << "'";
    }
  }
  return moves;
}

namespace {

class IdentityPolicy : public Policy {
public:
  StringRef name() const override { return "identity"; }
  FailureOr<LoopTree> run(const LoopTree &start, const Tuner &,
                          const Budget &, StringRef) override {
    return start;
  }
};

/// Applies the schedule given in `options` verbatim. Loop names refer to the
/// tree as it is when each move is applied.
class FixedPolicy : public Policy {
public:
  StringRef name() const override { return "fixed"; }
  FailureOr<LoopTree> run(const LoopTree &start, const Tuner &tuner,
                          const Budget &, StringRef options) override {
    LoopTree cur = start;
    auto moves = parseSchedule(options, cur);
    if (failed(moves))
      return failure();
    for (const Move &m : *moves) {
      std::string why;
      if (failed(tuner.isLegal(cur, m, &why)))
        return func::FuncOp(cur.func).emitError("schedule: illegal move ")
               << m.str(cur) << ": " << why;
      auto next = tuner.apply(cur, m);
      if (failed(next))
        return failure();
      cur = *next;
    }
    return cur;
  }
};

} // namespace

std::unique_ptr<Policy> createPolicy(StringRef name) {
  if (name == "identity")
    return std::make_unique<IdentityPolicy>();
  if (name == "fixed")
    return std::make_unique<FixedPolicy>();
  return nullptr;
}

SmallVector<StringRef> policyNames() { return {"identity", "fixed"}; }

} // namespace tune
} // namespace loom
