# Findings

## Repository
- Repository is on branch `master` and has an `origin`; no remote push is authorized.
- Existing implementation directories are empty; current tracked assets are GDS fixtures and a KLayout notebook.
- `TestReticle/gcd_45nm.gds` is untracked user data and must be preserved.

## Environment
- Canonical interpreter: `D:\app\miniforge\envs\myopc\python.exe` (Python 3.12.0).
- Installed: KLayout 0.30.10, NumPy 2.5.1, psutil 7.2.2.
- Missing at discovery: pytest and pytest-cov.
- Hardware: AMD Ryzen 7 4800H, 8 cores / 16 threads, 15.37 GB RAM.

## Existing GDS Baselines
- `simple.gds`: user-editable example, DBU 0.001 um, one top (`TOP`), one layer (1/0); its initial 10-shape baseline is historical and no longer used by deterministic tests.
- `JustPoly.gds`: one top, two polygons, negative coordinates.
- `test1.gds`: two tops (`test`, `cell1`), layers 2/0, 1/0, 3/0; `cell1` materializes 22, 13, and 1 polygon-like shapes respectively.
- A KLayout prototype queried 25 shapes from a one-million-instance AREF in about 0.3 ms and clipped them in about 2.3 ms on this machine.

## Architecture Constraints
- Never flatten the full layout by default.
- Never expose all polygons as Python lists.
- Use one lazy query per cell/layer/ROI and materialize only local Region batches.
- KLayout Region converts boxes and paths to polygons and ignores text/edges.
- Convert Region to compact NumPy contour/edge arrays only at the repeated-compute boundary.
- Source layout remains immutable; patch ownership is resolved in global DBU coordinates.

## Final Benchmark Baseline
- Million logical-instance AREF: 25 ROI polygons, 0.1058 ms median query+clip, 0.1164 ms p95, 0.48 MB RSS delta, 10.01 ms file open.
- 100,000-edge local grid: 425.48 ms one-time build, 0.0208 ms median indexed query versus 0.3325 ms brute scan, 15.99x speedup, exact results.
- 2,048 × 2,048 native grayscale raster: 482.84 ms, 4.32 MB RSS delta, exact aligned-area coverage.

## Real Layout Smoke Test
- Read-only validation of user-owned, untracked `TestReticle/gcd_45nm.gds` succeeded; the file was not added to Git.
- Selected top Cell: `TOP`; DBU: 0.0001 um; Layer: 11/0.
- Full top bbox: `[11400, 13150, 317300, 308850]` DBU.
- Materialized 1,776 polygons with total area 28,594,652,500 DBU².
- NumPy boundary conversion produced 21,590 vertices, 1,776 rings, and 21,590 closed edges.
- Optional diagnostics agreed on 1,776 polygon-like objects and reported zero Text/Edge/other objects.
- End-to-end direct CLI wall time measured by PowerShell was 461.001 ms, including interpreter and imports.

## Raster Feature Design
- Before phase 8, the code could query and clip planner ROIs and write GDS/OASIS patches, but had no pixel renderer or PNG output.
- KLayout 0.30.10 `Region.rasterize(origin, pixel_size, nx, ny)` returns exact covered area for each pixel as a bottom-up Python float matrix.
- Native rasterization does not apply merged semantics; the local Region must be merged first so overlaps are not counted twice.
- A PNG array must be vertically flipped because image row zero is top while KLayout raster row zero starts at the lower-left origin.
- Pillow and Matplotlib are installed; Pillow is sufficient for grayscale PNG save and optional system-viewer display without a GUI framework.
- `gcd_45nm.gds` full bbox is 305,900 × 295,700 DBU. At DBU 0.1 nm and 5 nm/pixel it becomes exactly 6,118 × 5,914 = 36,181,852 pixels.
- `.vscode/` and `TestReticle/gcd_45nm.gds` are user-owned untracked items and must remain outside Git commits.
- User confirmed the tracked `TestReticle/simple.gds` worktree change is intentional; preserve it and exclude it from this feature commit.
- Exact automated baselines now use generated temporary GDS files, so user edits to example reticles do not invalidate deterministic tests.
- Implemented two-dimensional native raster tiles, so the one-million temporary-pixel bound also holds for images wider than one million pixels.
- Targeted raster and direct CLI verification passed: 10 tests in 1.23 seconds; Ruff checks passed.
- Full real-layout raster succeeded for `gcd_45nm.gds` Layer 11/0 at 5 nm/pixel: 6,118 × 5,914 pixels, 4.663 seconds end-to-end, 149.36 MB peak process RSS, and a 78,464-byte PNG.
- Visual inspection confirmed top-up orientation and a clear white-mask/black-background rendering across the complete layout.
- The generated real-layout preview is an ignored test artifact at `.benchmarks/gcd_45nm_full_5nm.png`; source GDS and preview remain outside Git.

## MB-OPC Frontend Design
- OpenILT's useful ideas are corner-aware segmentation, fixed reference geometry, inner/outer sampling, maximum displacement, and core/halo separation; its list aliasing, repeated deep copies, pixel-probe normal inference, full-frame coupling and missing cross-tile identities are not suitable here.
- A GDS hole round trip can expose a ten-point keyhole hull with a zero-width bridge. KLayout `Region.merged()` converts it back to one valid hull plus one hole, so normalization must occur before contour extraction.
- `gcd_45nm.gds` currently contains 21,590 mathematical edges. The chosen 16/32 nm policy estimates 223,553 control segments.
- A read-only NumPy prototype measured 10.37 MiB for compact edge-ID/parameter/key storage versus 25.72 MiB for persistent expanded endpoints/normals/parents, a 59.7% reduction; index construction took about 4.2 ms.
- The largest Python-only speedups are one-time physical-boundary caching, edge-level metadata, lazy coordinate materialization, sorted stable-key lookup, sparse core membership and regular-grid point location.
- Physical-mask normalization, rectilinear core grids, line sampling and annotated boundary visualization are reusable by ILT evaluation and later OPC methods; displacement/reconstruction remain MB-OPC-specific.
- The common mask regression proved that minimum-coherence keeps corner-touch components separate, overlap cut-lines disappear, and a GDS keyhole becomes exactly one hull plus one hole with eight physical edges.
- The first real implementation run on `gcd_45nm.gds` produced 223,553 segments with 13.79 MiB persistent arrays, 290.01 ms preparation, 36.25 ms coordinate materialization, 960.54 ms full reconstruction, and zero XOR area.
- The 2x1 grid stored 226,813 sparse memberships for 223,553 segments, confirming that halo context scales with nearby cores instead of an S×C matrix.
- The first synthetic overlay was geometrically correct but only 276×266 pixels because DBU rasterization cannot use a sub-DBU pixel. At that size 71 segment labels obscured the mask, so diagnostic rendering needs a display-only nearest-neighbor enlargement plus independent label/sample decimation; this must not affect physical coordinates or reconstruction.
- Final storage audit found `fragment_indices` and `fragment_counts` duplicated information already encoded by `t0/t1` and stable keys. Removing them saves 8 bytes per segment (about 1.71 MiB on `gcd_45nm`) without adding iteration-time computation or reducing artifact reconstructability.
- The deterministic 110,000-segment benchmark measured 43.4% persistent-array savings versus a conservative fully expanded representation. The remaining sorted-key order/token arrays are intentional: removing them to chase the provisional 50% target would force every update round to rebuild lookup state, conflicting with iteration speed priority; the strict regression floor is therefore 40%.
- The corrected direct real-file flow prepares physical geometry while `LayoutDB` is open, then safely completes updates and artifacts after close. On `gcd_45nm` it used 12,675,300 persistent segment bytes, prepared in 282.89 ms and completed all artifacts in 2.951 s with zero reconstruction and stitch XOR.
- Visual inspection of the five-case atlas confirmed readable normals, owner colors, holes and four-core boundaries; the 1200-pixel `gcd_45nm` overlay clearly separates the two owner domains while decimated labels/samples preserve the full mask topology.

## Function Call Architecture Document
- The runtime has three distinct call phases: database-scoped preparation, database-independent iteration/reconstruction, and optional diagnostics/artifact output.
- `run_mbopc_frontend.run` is the orchestration root; `_prepare_input_problem` is only a CLI unit-conversion/grid adapter, while `prepare_problem` is the reusable library facade.
- `prepare_problem` fans out to `normalize_physical_mask`, `fragment_edges`, `OwnershipPolicy.assign`, and `build_sample_template`, then returns the immutable reference container `MBOPCProblem`.
- The iteration hot path is deliberately short: `merge_owner_updates` -> `SegmentBatch.lookup_keys`, followed by `SegmentBatch.materialize` -> `sample_lines`; no layout query, mask merge, contour extraction, edge fragmentation, hash construction or owner assignment repeats.
- Reconstruction is method-specific: `reconstruct_region` -> `reconstruct_contours` -> `SegmentBatch.materialize`/`validate_contours` -> `contours_to_region`.
- Output is a fan-out after reconstruction: `PatchSet.add/region`, `save_problem_npz`, `write_debug_gds`, `render_boundary_overlay`, and optional `run_geometry_suite`; none of these are required by a future solver loop.
- The shared-method boundary is `layout` + `geometry` + `opc.input`; edge-oriented construction lives in `opc.input.edge`, while future concrete iteration methods depend downward from `opc.iteration.<method>`.
- The delivered document contains 453 lines, nine Mermaid blocks and 19 verified relative source links; no link target is missing and all Markdown fences are balanced.
- Mermaid CLI is not installed in the current environment, so validation is structural rather than a local SVG render; the diagrams use only standard `flowchart` syntax, quoted labels and subgraphs/arrows supported by common Markdown Mermaid renderers.
- `.vscode/launch.json` has a concurrent user change adding an MB-OPC debug configuration; it is unrelated to this documentation task and must remain unstaged.

## OPC Directory Responsibility Split
- The repository has no concrete iterative solver yet: the former `opc.mbopc` files construct edge-oriented input, assign ownership, carry updates, reconstruct contours and emit diagnostics.
- The move-only target is therefore `opc.input` for method-neutral input data and `opc.input.edge` for edge-segment input mechanics; future solvers belong under `opc.iteration.<method>`.
- `lithography/`, `evaluation/`, `opc/iteration/` and `opc/iteration/mbopc/` may be created as empty physical directories, but Git cannot retain them until a real implementation file is added; adding placeholder files would violate the user's no-new-file constraint.
- The user-owned `.vscode/launch.json` change and untracked `gcd_45nm.png` must remain untouched and unstaged; `layout/` and `geometry/` must have no content diff.
- Post-move verification passed: 37 OPC tests at 93% combined statement/branch coverage, 81 full-repository tests, scoped Ruff and compileall.
- AST comparison of the 12 non-initializer production modules found zero logic mismatch after removing imports and module docstrings; the two initializers only changed exports and names.
- Both obsolete import specs (`opc.common`, `opc.mbopc`) now resolve to `None`; all 19 source links in the call-architecture document resolve after the move.

## Physical Tile-Size CLI
- Existing `--grid COLUMNS ROWS` divides the selected processing box, not unconditionally the full reticle; without `--box` that processing box is the selected top-cell bbox.
- A grid core is the tile's unique ownership/output region. Its `halo_nm` is not another output tile: the halo is read-only context used by nearby segment membership.
- The new mode should be `--tile-size-nm SIZE`, mutually exclusive with an explicitly supplied `--grid`; it creates square fixed-size cuts anchored at the selected box's left/bottom and clips the final column/row to the right/top boundary.
- Unit conversion should remain in the root CLI adapter because `RectilinearCoreGrid` intentionally stores integer DBU cuts and is shared by callers that may not use nanometers.
- The first 100 nm synthetic run generated exact cuts `x=[-20,80,180,250]`, `y=[-20,80,180,240]`, but nine per-core KLayout intersections did not reconstruct the diagonal-containing Region exactly.
- The discrepancy exists at zero displacement (35 DBU² XOR) and after demo updates (29 DBU²), so it is not caused by the new physical-size conversion or update policy.
- Direct manual per-box intersections and `PatchSet` produce identical loss; merging the fragments does not repair it. The failure boundary is the existing integer Region clipping of diagonal polygons across multiple ownership boxes.
- Completing a trustworthy whole-reticle tile mode therefore requires a focused `geometry/patch.py` correction plus diagonal/multi-row/multi-column regressions. Skipping the stitch check in the runner would hide data loss and is rejected.
- The user approved the corrected responsibility split: core boxes partition computation and ownership, while canonical vector output uses one global reconstruction. This removes the unnecessary diagonal clipping operation rather than weakening its XOR assertion, and requires no `layout/` or `geometry/` change.
- `run_mbopc_frontend.run()` and its physical-tile regression are the only existing functions explicitly approved for this semantic correction; subsequent solver work should use new modules unless separately authorized.

## ICCAD13 Lithography And Evaluation
- The four actually used OpenILT assets total about 474 KiB: focus/defocus kernels are 24×35×35 and each scale vector has 24 float weights. Asset hashes are locked by tests and the MIT license is copied beside the implementation.
- The migrated aerial-image calculation matches OpenILT pixel-for-pixel on the real 200×150 patch shape for nominal/max/min. OpenILT's full 256×256 path incorrectly returns the input mask from `unpad`; MyOPC deliberately fixes that and has a regression proving full-canvas output remains continuous.
- Direct `myopc/python.exe` on Windows needs `<env>/bin` both registered through `os.add_dll_directory()` and prepended to the process `PATH`; the latter is required because NVRTC performs its own builtins lookup. A direct CUDA subprocess test now covers this without `conda run`.
- Two 256×256 masks used about 64.45 MiB peak CUDA allocations in the first smoke test. The model remains a differentiable `torch.nn.Module`; MB-OPC will wrap calls in `torch.no_grad()`, while future ILT can reuse gradients.
- Evaluation keeps images on their existing device, excludes halo pixels with an ownership mask, and returns only scalar L2/PVBand plus compact per-segment EPE directions. Invalid narrow-feature/out-of-bounds/same-pixel probes never move an edge; simultaneous inner/outer violations are explicitly ambiguous with direction zero.

## Streaming Simple MB-OPC
- `opc.iteration.mbopc.optimize` keeps only the compact global displacement/contour state on CPU. It creates current/target/ownership images for one batch, returns only scalar metrics and compact owner directions, then releases the batch tensors.
- Every tile in a round reads `current`; owner updates accumulate in `next_values` and become visible only after all batches complete and global contour validation succeeds. Cross-core order therefore cannot update an edge early.
- Target tiles are cached as uint8 under an explicit LRU byte bound. A regression caught and fixed a cache-hit normalization error that would otherwise pass values up to 255 into the second round.
- Final vector output is reconstructed once from global best displacements. Core boxes control evaluation ownership and update authority but never clip final polygons.
- Candidate topology safety is deliberately conservative in v1: any predictable reconstruction failure rolls back the complete round. This avoids partial-polygon publication without adding an unproven per-polygon recovery layer.
- The 2 DBU hollow-wall/8 DBU probe case invalidates long-edge probes through target semantics. A few corner fragments can remain locally valid because their inward normal crosses the adjacent perpendicular wall; tests distinguish this geometry rather than incorrectly declaring every probe invalid.
- Direct `simple.gds` CUDA validation used 8 cores and 885 segments. Three rounds reduced EPE 338 → 203 → 113, used about 65 MiB peak allocated GPU memory, and produced valid GDS/NPZ/JSON/PNG artifacts.
- The new `run_mbopc.py` is directly executable from any working directory without installing MyOPC. Defaults target `TestReticle/simple.gds`; full-reticle and ROI runs share fixed-nm tile, halo, pixel-grid and batch controls.
- Final-code full `gcd_45nm` validation processed 870 cores, 223,553 segments and 880,801 memberships in three rounds. EPE fell 129,645 → 74,592 → 48,348 and L2 fell 1,038,629.522 → 563,335.522 → 440,251.431; PVBand rose and is reported separately rather than hidden.
- The final run took 84.708 seconds, peaked at 271,544,320 CUDA-allocated bytes on a 4 GiB GTX 1650, and wrote valid GDS/NPZ/JSON/PNG artifacts. This validates bounded streaming behavior but is not a maximum-batch claim for the target 24 GiB GPU.
- Topology regressions proved plain reconstruction accepted both a rectangle whose left edge crossed the right edge and a hull moved inside its hole. The solver now rejects ring-orientation changes and hole escape before the round barrier; legal `gcd_45nm` results remained bitwise-identical in metrics.

## OPC Architecture Subtraction Audit
- The complete repository baseline is healthy: 119 tests passed in 19.26 seconds before phase 25 changes.
- `problem.physical_mask.contours/edges` and `problem.segments.contours/edges` are immutable shared references rather than copied geometry arrays; retaining those aliases keeps the two domain objects self-contained without material memory cost.
- Stable `edge_keys`, `segment keys`, `_key_order` and `_sorted_tokens` consume 7,499,136 bytes on `gcd_45nm`, 59.16% of the reported 12,675,300-byte `SegmentBatch` persistent storage.
- The current streaming solver bypasses stable keys and `merge_owner_updates`; those paths are sustained only by the frontend demo, tests, benchmark and NPZ export. The user selected complete removal and current-process index identity.
- `edge_segment_offsets` has no production consumer outside diagnostic export, and persistent `edge_lengths` only feeds materialized diagnostic lengths. Both can be derived only when explicitly requested.
- `BoundarySampleTemplate` is stored on every problem although the solver computes EPE probes independently. Its default normal offset comes from `corner_length`, while the solver uses `epe_distance`; separate values make the preview disagree with actual evaluation.
- The user selected regular `RectilinearCoreGrid` ownership only and requested that `run_mbopc_frontend.py` remain as a separate direct verifier.
- NPZ remains only in the explicitly invoked frontend verifier, becomes index-aligned without keys, and is no longer emitted by the full solver entry point.
- `UniformGridIndex` and the replaceable-backend claims around `GeometryEngine` remain separate protected-directory audit findings; phases 25-27 must not change `layout/` or `geometry/`.
- `SegmentGeometry.lengths` and `segment_indices` were allocated on every full materialization although reconstruction consumes only starts/ends/normals; diagnostic maximum lengths can be derived from `np.linalg.norm(ends-starts)` only when requested.
- The three edge-input files `artifacts.py`, `visualize.py` and `verification.py` are all explicit diagnostic lifecycle code. Consolidating them under `opc/diagnostics.py` removes output/test responsibilities from input construction without inventing a new directory.
- Full `run_mbopc.py` tests currently assert a default NPZ; this must be deliberately updated so only its GDS/JSON remain mandatory, while the retained frontend verifier checks NPZ format version 2 without key arrays.
- After core source/test migration, the only remaining Python references to removed key/length APIs are in `benchmarks/benchmark_mbopc_frontend.py`; documentation references are intentionally deferred to the synchronized report phase.
- `opc.diagnostics` now owns every explicit artifact and geometry-atlas caller. `opc.input.edge` no longer imports or exports diagnostics, preserving one-way dependency and avoiding an import cycle.
- The post-migration Python call-site search finds no remaining use of stable keys, external update batches, template sampling, selective materialization, persistent edge offsets/lengths, or the old reconstruction signature.
- The source diff currently deletes four obsolete edge-input modules, adds one diagnostics module and leaves all protected `layout/`/`geometry/` files untouched.
- Post-refactor full regression passes 118 tests in 15.31 seconds; the count changed because tests dedicated only to deleted APIs were removed while new alias/probe regressions were added.
- The strict 5,000-shape/110,000-segment benchmark passes: 120.11 ms prepare, 11.84 ms materialize, 441.61 ms zero reconstruction, 2.441 MiB persistent segment arrays and 69.38% saving versus expanded storage.
- The historical same-scale benchmark was 168.41 ms prepare, 17.04 ms materialize, 477.95 ms reconstruction and 43.43% saving, so the subtraction improves every measured hot-path time while substantially lowering persistent storage.
- Real `gcd_45nm` frontend preparation remains exact with 223,553 segments: persistent arrays fell from 12,675,300 to 4,830,716 bytes (61.89% reduction), prepare took 161.12 ms, total diagnostic run 2.357 s and zero-displacement XOR remained zero.
- Visual inspection of the regenerated whole-layout overlay confirms complete red/cyan owner partitioning at the vertical core boundary, continuous cross-core geometry, readable labels and paired cyan-inner/red-outer probe markers.
- Focused coverage verification passes 74 tests with 92% combined statement/branch coverage for `opc`, `lithography` and `evaluation`.
- Three-round full `gcd_45nm` CUDA results are numerically identical to the pre-refactor baseline: EPE 129,645→74,592→48,348, L2 1,038,629.522→563,335.522→440,251.431 and peak allocation 271,544,320 bytes.
- Full-flow total time is 85.473 seconds versus the historical 84.708 seconds (0.90% difference, normal run variance); frontend preparation is 0.249 seconds and persistent segment arrays are 4,830,716 bytes.
- The full runner artifact schema now contains only summary JSON, result GDS and optional preview PNG; no NPZ path or file is produced.
- 当前开发手册、测试手册、函数调用文档和两组专项报告仍保留稳定键、外部更新批次、可替换 owner 策略及全流程 NPZ 等旧描述，必须按实际代码整体同步，不能通过局部术语替换掩盖数据流变化。
- `doc/design_review.md` 和 `CLAUDE.md` 均为已跟踪的历史/用户文档；本阶段不改写设计审查历史，也不让旧兼容说明扩大本次 OPC 实现范围。
- 开发手册与测试手册已按当前数组下标身份、单一规则网格、固定参考同步迭代和诊断输出边界重写；其中明确记录 PVBand 上升，避免只展示改善指标。
- 函数调用文档已按实际入口、输入构造、浅引用、同步轮次、探针、全局重建、诊断边界和扩展位置重建；旧专项报告仍是剩余的主要陈旧引用来源。
- 五份既有 MB-OPC 专项文档已全部按当前实现重写，并新增架构精简开发/测试报告；稳定 key、外部更新和全流程 NPZ 只作为“已删除/不支持”历史对照出现。
- 当前 `run_geometry_suite` 重新生成 5 个图形案例，全部零位移 XOR 为 0；segment 数分别为 56、70、39、78、48，图与 JSON 均来自精简后的同一诊断实现。
- 人工查看孔洞与整张 `gcd_45nm` 图：孔洞内外 probe 方向、四 core 边界和左右 owner 分区正确，跨 core 轮廓连续；无显示层面的回归。
- 最终静态审计覆盖 76 个第一方 Python 文件：中文模块/函数 docstring 零缺失；13 份 `doc/*.md` 无断链或不平衡代码围栏；生产模块没有零引用函数。
- 删除符号搜索只命中两个刻意断言旧字段不存在的回归测试；Ruff、compileall 和全仓库 118 项测试最终复跑通过（19.29 s）。
- 最终覆盖率复核为 74 项通过（15.94 s）、OPC/光刻/评价综合语句/分支覆盖率 92%，`opc.iteration.mbopc.solver` 本身为 92%。
- 用户现已授权修改 `layout/`、`geometry/` 以继续精简；此前保留这些目录不是遗漏，而是遵守逐次确认边界。当前调用搜索显示 `UniformGridIndex` 只有测试/基准调用，`RegionBatch.backend` 永远固定为 `klayout`，`GeometryEngine` 是固定后端上的无状态包装，三者需要进一步逐项审计。
- 历史设计评审同样把 `UniformGridIndex` 判为零生产调用方，把 `GeometryEngine` 的多后端理由判为不成立；当前完整流程已经稳定，满足其“真实工作流稳定后再退役”的前提。
- `ShapeQuery.materialize()` 当前只用 ROI 限定候选，完整 runner 额外调用 `GeometryEngine.clip`，但 MB-OPC 直接消费者没有同一步骤；把精确 Region 相交统一到物化边界可以同时删除门面唯一生产调用并修复不同入口的 ROI 语义差异。
- 阶段 28 已删除 `geometry/region.py`、`geometry/spatial.py`、固定 backend 字段/异常、edge bbox 属性及仅服务这些 API 的两份测试；direct runner 与层级基准改为消费精确物化结果。
- 新回归用 100×100 DBU 图形查询内部 50×60 DBU ROI，要求物化 bbox 精确等于查询框且面积为 3,000 DBU²，直接覆盖此前不同入口裁剪语义不一致的问题。
- KLayout 普通 Region `&` 会丢弃属性；本地 API 实验证实 `and_(clip, NoPropertyConstraint)` 保留左侧 Polygon 属性。属性模式现使用该原生批量约束，普通模式继续使用更直接的 `&`。
- 属性回归现在让 ROI 同时截断普通/带属性两个图形，证明精确裁剪后数量一致且 tagged 属性仍准确；修复后 39 项 layout/geometry/OPC 聚焦测试与 Ruff 通过。
- 精简后的严格 Layout/Geometry 基准通过：百万实例 ROI 精确物化中位数 0.10435 ms、P95 0.16723 ms、RSS 增量 0.484 MiB；2048² 栅格 499.59 ms、6.379 MiB、覆盖完全一致。
- 活跃源码已无 `GeometryEngine`、`UniformGridIndex`、backend 标记或 edge bbox 调用；剩余陈旧引用仅在两份 Layout/Geometry 报告中，历史 `design_review.md` 作为当时审计证据保留不改。
- Layout/Geometry 专项 38 项通过（1.49 s），综合 statement/branch coverage 91%；删除 5 项仅服务旧 API 的测试、新增 1 项生产 ROI 回归，不降低现有路径覆盖。
- 全仓库最终静态/语法/回归初次复跑通过：Ruff、compileall、114 tests passed in 15.74 s。测试数由 118 净减 4，恰好对应删除 5 个旧门面/索引用例并新增 1 个 ROI 回归。
- 阶段 28 真实 `gcd_45nm` 前端复跑保持 1,776 polygons、21,590 edges、223,553 segments、4,830,716 persistent bytes 和全部零面积差异；prepare 152.82 ms，总计 2.308 s。
- 三轮 CUDA 完整复跑逐项指标再次完全一致：EPE 129645→74592→48348，L2 1038629.522→563335.522→440251.431，PVBand 115626.751→134540.869→147186.806，GPU 峰值 271,544,320 bytes；总计 85.892 s，结果合法且无 NPZ。
- 阶段 28 最终 MB 前端严格基准仍全部通过：prepare 125.64 ms、materialize 12.45 ms、zero reconstruct 427.83 ms、persistent 2.441 MiB、内存节省 69.38%、XOR/unowned/strict failures 均为 0。
- 最终引用审计另发现 `DbuBox.overlaps` 只是 `intersection(...) is not None` 的无调用方包装，已删除并让原测试直接断言 intersection 语义。
- `LayoutDB.hierarchy_summary` 虽无内部生产调用，但它是早期明确交付的只读层级检查公共能力，且没有第二套实现或热路径常驻成本；保留它不是为假设后端建立的抽象。
- 最终代码复跑通过：Ruff、compileall、全仓库 114 tests（16.03 s）；Layout/Geometry 38 tests（1.50 s）、综合 statement/branch coverage 91%。
- 最终静态审计覆盖 72 个第一方 Python 文件和 13 份 Markdown：中文 docstring 零缺失、链接/围栏零错误、删除符号零生产引用、diff whitespace 零错误；用户 `CLAUDE.md`、历史 `design_review.md` 与 `TestReticle` 无差异。
- 零内部生产引用只剩 `hierarchy_summary` 与 `DbuBox.intersection` 两个明确公共基础能力；前者提供只读层级快照，后者提供 planner DBU 框精确关系，均实现紧凑、无并列抽象且已有直接测试，因此有意保留。
- 阶段 29 用户只授权两项优化：ownership 构建避免完整 `SegmentGeometry` 临时物化；零位移初态和局部未变化 core 跳过轮廓/Region 重建。稀疏活跃 core 方案暂缓，不得增加 `core_at`、改变 core 数量或修改分块 JSON 语义。
- 11 万 segment 的 ownership 等价对照中，owner/core offsets/memberships 逐项一致；中位耗时 `40.20 -> 37.15 ms`，tracemalloc 峰值 `24.56 -> 17.85 MiB`。端点中间表必须在 CSR 展开前显式释放，否则 Python 局部引用会抵消省掉 normals 的内存收益。
- target LRU 使用 uint8 常驻，而历史 current mask 保留浮点像素覆盖率；零位移 core 因此不能直接返回 target。当前快路直接栅格化参考 Region，只跳过轮廓子集与 Region 差分，保持原评价数值语义。
- 零位移局部等价微基准（500 polygons/11,000 segments）像素逐项相同，中位耗时 `44.35 -> 11.06 ms`；`gcd_45nm` 三轮 CUDA 的 EPE/L2/PVBand 等全部指标与阶段 28 逐项一致，总耗时 `85.892 -> 79.117 s`，GPU 峰值仍为 271,544,320 bytes。
- 阶段 29 最终审计覆盖 72 个第一方 Python 文件和 20 份 Markdown：中文模块/函数 docstring、链接、代码围栏、diff whitespace 均无问题；没有新增生产抽象、死函数、重复字段或异常兼容分支，`layout/`、`geometry/` 无差异。
