"""CSV exports for native benchmark reports."""
import csv
import json


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
             for key, value in row.items()}
            for row in rows
        )
