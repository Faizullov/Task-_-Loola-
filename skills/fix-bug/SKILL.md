---
name: fix-bug
description: Reproduce a coding defect, follow the failing path, and verify a minimal fix.
metadata:
  tools: search_text read_file apply_patch run_checks git_diff
---

1. Find the failing path and run the smallest existing check that can reproduce the defect.
2. Read the implementation and describe the cause using the observed output.
3. Add a focused regression test when the behavior can be tested, then apply the minimal fix.
4. Run the failing check again, review the diff, and report what the checks confirmed.
