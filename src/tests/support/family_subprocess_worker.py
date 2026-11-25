"""Local subprocess evaluation fixture; no API calls or scientific fitting."""
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
for key in ('family', 'task', 'output', 'config', 'reference-cache', 'reference-mode'):
    parser.add_argument('--' + key)
args = parser.parse_args()
task = json.loads(Path(args.task).read_text())
config = json.loads(Path(args.config).read_text())
time.sleep(config.get('sleep', 0.05))
if task['case_id'] == 'error':
    sys.exit(7)
result = dict(status='complete', context_id=task['context_id'],
              case_id=task['case_id'], gap_pct=1.25, runtime_seconds=0.05)
if task['case_id'] == 'wrong-case':
    result['case_id'] = 'incorrect'
if task['case_id'] == 'nonfinite':
    result['gap_pct'] = float('inf')
Path(args.output).write_text(json.dumps(result))
