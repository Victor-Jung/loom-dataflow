/// pybind11 bindings for the Loom MLIR pipeline.
///
/// Exposes two pipeline functions and a version string:
///   - run_exploration_pipeline(...)   → stages 0-5
///   - run_materialization_pipeline(...) → stages 5-7
///   - run_mapping_tune_pipeline(...)    → stage 0 → tuned stage 0
///   - __version__                     → compile-time version from CMake

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loom_version.h"
#include "loom_exploration_pipeline.h"
#include "loom_materialization_pipeline.h"
#include "loom_mapping_tune_pipeline.h"

namespace py = pybind11;

PYBIND11_MODULE(_loom_pipeline, m) {
  m.doc() = "Loom MLIR pipeline - pybind11 interface";
  m.attr("__version__") = LOOM_VERSION_STRING;

  m.def(
      "run_exploration_pipeline",
      &loom::pipeline::runExplorationPipeline,
      py::arg("input_mlir_text"),
      py::arg("hw_spec_file"),
      py::arg("produce_etg") = true,
      py::arg("skip_etg") = false,
      py::arg("full_occ") = false,
      py::arg("spatial_reuse") = true,
      R"doc(Run the exploration pipeline (stages 0-5).

      Consolidates tensor_canonicalize, memory_binding, enumerate_hw_mapping,
      analyze_reuse, and enumerate_copy_broadcast into a single in-memory run.

      Args:
          input_mlir_text: Input MLIR as a string (stage 00).
          hw_spec_file: Path to hardware specification MLIR file containing
              hardware description and compute/data mover components.
          produce_etg: Whether to produce ETG JSON (default True).
          skip_etg: When True, skip staged ETG generation.
          full_occ: When True, use only full hardware occupancy.
          spatial_reuse: When True, run reuse analysis and copy/broadcast
              enumeration.

      Returns:
          Tuple of (error, output_mlir, etg_json).
          error is empty on success; etg_json is empty when ETG is skipped.
      )doc",
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "run_materialization_pipeline",
      &loom::pipeline::runMaterializationPipeline,
      py::arg("input_mlir_text"),
      py::arg("block_sizes_json"),
      R"doc(Run the materialization pipeline (Materialize -> OSB).

      Takes explored MLIR and block sizes from the external solver, materializes
      symbolic values, canonicalizes, and runs One-Shot Bufferization.

      Args:
          input_mlir_text: Input MLIR as a string (stage 05).
          block_sizes_json: JSON string mapping variant names to one or more
              block-size assignments.

      Returns:
          Tuple of (error, output_mlir).
          error is empty on success.
      )doc",
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "run_mapping_tune_pipeline",
      &loom::pipeline::runMappingTunePipeline,
      py::arg("input_mlir_text"),
      py::arg("policy") = "identity",
      py::arg("options") = "",
      R"doc(Tune a mapping program (stage 00 MLIR) with a search policy.

      Runs the loom-tune-mapping-program pass on every function and returns
      a stage 00 program accepted by run_exploration_pipeline.

      Args:
          input_mlir_text: Input MLIR as a string (stage 00).
          policy: Search policy name ("identity", "fixed", ...).
          options: Policy options; for "fixed" the schedule string, e.g.
              "interchange(m,n)".

      Returns:
          Tuple of (error, output_mlir). error is empty on success.
      )doc",
      py::call_guard<py::gil_scoped_release>());
}
