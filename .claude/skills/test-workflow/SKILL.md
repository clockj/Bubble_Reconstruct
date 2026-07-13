---
name: test-workflow
description: When writing or running test/validation scripts in visual_hull/, ensure test outputs follow the project convention — timestamped subfolders under test/, no overwrites, and a report.md or report.json at the end.
directory: visual_hull
---

# Test Workflow Convention

When creating or running any test, validation, benchmark, or diagnostic script under `visual_hull/`, follow these rules:

## Output Location

- All test results, logs, screenshots, and generated files **MUST** be saved under the `visual_hull/test/` folder.
- Each new test run **MUST** create a new subfolder — never overwrite a previous run.
- Subfolder naming: `YYYYMMDD-HHMMSS-<short-name>` (e.g., `20260713-143022-smoke-test`).

## Required Report

After a test run finishes, generate a report and save it into that run's subfolder:

- `report.md` for human-readable summaries
- `report.json` if structured/automated output is needed

## Script Behavior

Any script that runs tests should:

1. **Create** the run subfolder at the start of execution
2. **Write** all artifacts (logs, plots, exports, etc.) into that folder
3. **Write** the final report (`report.md` or `report.json`) into that folder at the end

## Helper

Use `visual_hull.test_runs.create_test_run(project_root, test_name)` to scaffold the timestamped folder automatically. It returns a `TestRun` object with `write_json()` and `write_text()` methods.
