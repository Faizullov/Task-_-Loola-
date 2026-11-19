---
name: verify-changes
description: Review a patch and select appropriate syntax and test checks before handing it off.
metadata:
  tools: git_diff run_checks
---

1. Inspect the diff to identify changed Python files and behavior. Include newly created files in the review.
2. Run syntax checks for changed Python entry points and targeted tests for changed behavior.
3. Run the full test suite when changes affect shared behavior.
4. Report the commands or check names, their outcomes, and any checks that could not run.
