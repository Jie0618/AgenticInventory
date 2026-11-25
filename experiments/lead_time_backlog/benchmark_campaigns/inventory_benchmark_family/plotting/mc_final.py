"""Plot the completed cyclic MC campaign using the comparison layout."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reporting.plots import compare
CODEX_ID = "codex-astra-urgency-correction"
COLOR = "#8C564B"
LABEL = "Codex-gpt-6-astra"

BASELINES = ['constant_order', 'base_stock', 'capped_base_stock', 'sq', 'qr', 'gbs', 'myopic', CODEX_ID]
# Policies shown in the comparison.
OURS = ['BATS', 'CBATS']
NON_SOF = ['myopic']
LABELS = {'constant_order': 'Constant Order', 'base_stock': 'Base Stock', 'capped_base_stock': 'CBS', 'sq': 'SQ', 'qr': 'QR', 'gbs': 'GBS', 'myopic': 'Myopic', 'BATS': 'BATS', 'CBATS': 'CBATS'}
DEFAULT_RUN = Path(__file__).resolve().parents[1] / "runs/mc_final"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_root / "figures"
    compare(args.run_root, output, BASELINES, OURS, 'K',
            labels={**LABELS, CODEX_ID: LABEL}, non_sof=NON_SOF,
            formats=("png",),
            policy_colors={CODEX_ID: COLOR}, overall_log_left=.8)
    print(output)


if __name__ == "__main__":
    main()
