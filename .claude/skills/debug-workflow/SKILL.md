---
name: debug-workflow
description: Plan-first debugging and testing workflow. Use when debugging, writing test/validation scripts, or investigating a bug — plan and get approval before implementing, keep all results next to the test code, and finish with a report.
---

# Debug Workflow

A plan-first loop for debugging and testing. Get approval before writing code, keep every
artifact beside the test script, and close with a report.

## Standard workflow

1. **Plan first.** Outline the approach and show it before writing any code.
2. **Wait for approval.** Do not implement until the plan is approved. Save the plan to the
   testing folder.
3. **Stay scoped.** While building the test code, do not run or modify unrelated code.
4. **Keep results local.** Save every result — logs, plots, data — to the same folder as the
   test code.
5. **Report.** End with a final report saved to the testing folder.

## File organization

- Save test results in the **same directory** as the test scripts.
- Use descriptive filenames, with timestamps where helpful.
- Never write results to a separate output folder.

## Code modification

- Explain each change **before** implementing it.
- Wait for confirmation before moving to the next step.
