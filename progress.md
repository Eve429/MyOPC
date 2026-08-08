# Progress Log

## 2026-08-09
- Recovered design context from the saved architecture HTML.
- Inspected repository, Conda environment, notebook, existing GDS files, hardware, and KLayout APIs.
- Confirmed the high-performance hierarchical ROI approach with an in-memory million-instance prototype.
- Locked user choices: KLayout backend, immutable source plus patches, complete foundation MVP.
- Added requirements for local milestone commits, compact documented code, and cross-core polygon patch testing.
- Phase 1 started.
- Added package metadata, documented Layout contracts, LayoutDB, hierarchy inspection, lazy ROI queries, and initial tests.
- Diagnosed KLayout ROI Region count behavior and added native shape flags for Box/Path/Polygon; no per-shape Python filtering added.
- Layout milestone verification: 12 tests passed; layout package coverage is 90%.
- First local milestone committed as `dbf44ff`.
- Implemented Geometry Region operations, contour/edge batches, validation, local grid index, patch ownership, and atomic patch writer.
- Geometry test collection initially failed because its directory lacked a package marker; fixed test packaging before functional rerun.
- Geometry functional rerun: 14 passed, 1 expectation failed because KLayout Region area uses set semantics even for two raw unmerged polygons; corrected the test to assert raw count and merge state.
- Added generated GDS/OASIS coverage for multi-layer hierarchy, Path/Text, holes, R90, mirror, and AREF.
- Corrected generated fixture expectations: layout bbox includes Text while Region materialization intentionally does not.
- Fixed optional diagnostics for hierarchical zero-area Text: diagnostic iterator uses touching semantics while the production polygon query remains area-overlap-only.
- User tightened delivery requirements: all comments/docstrings must be Chinese and more detailed at critical paths; add a direct Python entry point; finish with an explicit overdesign/error-complexity audit.
- Translated source/test comments and docstrings to Chinese and expanded critical-path explanations.
- Added `run_layout_geometry.py` for direct inspect/query/array/Patch execution without installing the project package.
- Simplified CLI Box input to four integers after comma-separated negative coordinates conflicted with argparse option parsing.
- Added direct-run performance benchmark with strict acceptance gates.
- Strict benchmark passed: million-instance ROI median 0.116 ms and 0.48 MB RSS delta; 100k-edge index queries were exact and 20.73x faster than brute-force bbox scans.
- Removed the editable project installation and reran verification: 31 tests passed, 91% coverage, compileall passed, and external-working-directory CLI execution passed.
