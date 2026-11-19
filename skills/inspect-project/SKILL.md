---
name: inspect-project
description: Find a project's entry points, implementation, and related tests before explaining or changing code.
metadata:
  tools: list_files search_text read_file
---

1. List the files near the likely entry point or use a filename pattern to narrow the tree.
2. Search for the specific symbol, configuration key, or error text from the task.
3. Read the surrounding lines and follow the relevant call sites into implementation and tests.
4. Report what the code does with file paths and line numbers. Record open questions when the available files do not settle them.
