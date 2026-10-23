# Validation status for v1.0.0

The release identity validator was executed after package construction and passed for all seven datasets, 16 model labels, and 47,952 frozen OOF rows.

The unified smoke-test runner was executed on 22 October 2026 in the manuscript's Spark PyTorch environment (`Python 3.10`, `PyTorch 2.12.0.dev20260408+cu128`). The following checks passed:

- attention semantics and cross-sample invariance;
- DEM-Fix cross-sample invariance;
- TRACE-GS DNA synthetic token allocation, forward/backward execution, and batch invariance;
- TRACE-GS RNA batch invariance and forward/backward execution.

Terminal marker: `ALL_SMOKE_TESTS_PASSED`.

The smoke tests check implementation invariants on synthetic inputs. They do not replace reproduction of the reported seven-trait experiments from the third-party biological datasets.

