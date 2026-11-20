import json


def decode_rows(lines):
    """Return event IDs from JSON lines."""
    return [json.loads(line)["id"] for line in lines if line.strip()]
