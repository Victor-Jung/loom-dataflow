#ifndef LOOM_MAPPING_TUNE_PIPELINE_H
#define LOOM_MAPPING_TUNE_PIPELINE_H

#include <string>
#include <tuple>

namespace loom {
namespace pipeline {

/// Tune a mapping program (stage 00 MLIR) in memory.
///
/// Runs the `loom-tune-mapping-program` pass with the given search policy on
/// every function of the module and returns the tuned module, which is again
/// a stage 00 program accepted by the exploration pipeline.
///
/// @param input_mlir_text  Stage 00 MLIR text.
/// @param policy           Search policy name ("identity", "fixed", ...).
/// @param options          Policy options; for "fixed" the schedule string.
/// @return tuple of (error, output_mlir). error is empty on success.
std::tuple<std::string, std::string>
runMappingTunePipeline(const std::string &input_mlir_text,
                       const std::string &policy,
                       const std::string &options);

} // namespace pipeline
} // namespace loom

#endif // LOOM_MAPPING_TUNE_PIPELINE_H
