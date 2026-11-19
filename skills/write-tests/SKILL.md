---
name: write-tests
description: Add focused Python tests for behavior or regressions using the project's conventions.
metadata:
  tools: search_text read_file apply_patch run_checks
---

1. Find existing tests and read the public entry point for the behavior.
2. Identify the expected result and the boundary the test should exercise.
3. Add a focused test with project conventions. Use unittest when no test framework is established.
4. Run the targeted test first and confirm the result. If broader checks are relevant, run the suite.
