# Plugin migration contract

Every plugins/loader_NN.py exports decode_rows(lines).
It returns IDs from JSON lines in their original order.
Skip blank lines and malformed JSON. Keep duplicate IDs.
A valid JSON object without an id is skipped.
An id can be a string or integer, including 0. Preserve its type.
Return [] for empty input.
All twelve plugins must implement this contract.
Add unittest regression coverage.
