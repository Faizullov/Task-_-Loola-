"""Verbose integration checks for one loader."""
import importlib
import sys

number = int(sys.argv[1])
module = importlib.import_module(f"plugins.loader_{number:02d}")
print(f"Checking plugins/loader_{number:02d}.py", flush=True)
print("CONTRACT: skip malformed JSON, blank lines, and objects without id. Preserve ID order, duplicates, integer zero, and ID types.", flush=True)
for index in range(80):
    print(
        f"TRACE case={index:03d} plugin=loader_{number:02d} "
        'input={"id":0} expected_id=0 expected_type=int '
        "operation=decode stage=contract_validation source=integration_fixture",
        flush=True,
    )
cases = [
    ([], []),
    (['', ' ', '{"id": "a"}'], ['a']),
    (['{"id": "a"}', 'broken', '{"id": "b"}'], ['a', 'b']),
    (['{"id": 0}', '{"id": 2}', '{"id": 0}', '{}'], [0, 2, 0]),
]
failed = 0
for lines, expected in cases:
    try:
        actual = module.decode_rows(lines)
        assert actual == expected, (actual, expected)
    except Exception as error:
        failed += 1
        print(f"FAIL input={lines!r} expected={expected!r}: {type(error).__name__}: {error}")
print(f"RESULT plugins/loader_{number:02d}.py: {len(cases)-failed}/{len(cases)} passed")
raise SystemExit(bool(failed))
