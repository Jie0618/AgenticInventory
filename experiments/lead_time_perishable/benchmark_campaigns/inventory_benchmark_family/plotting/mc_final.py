"""Plot the completed cyclic MC campaign using the comparison layout."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from reporting.plots import compare
CODEX_ID = "codex-astra-capped-survival"
COLOR = "#8C564B"
LABEL = "Codex-gpt-6-astra"

BASELINES = ['constant_order', 'base_stock', 'copsq', 'capped_base_stock', 'bsp_low', 'weighted_base_stock', 'bsp_2d', 'bsp_oir', 'ewa', 'bsp_low_ew', 'pil', 'apil', 'sqmax_ew', 'bsp_ws3', CODEX_ID]
OURS = ['SCBS', 'ETIS']
NON_SOF = ['ewa', 'bsp_low_ew', 'pil', 'apil', 'sqmax_ew', 'bsp_ws3']
LABELS = {'constant_order': 'Constant Order', 'base_stock': 'Base Stock', 'capped_base_stock': 'CBS', 'weighted_base_stock': 'WBS', 'bsp_low': 'BSP-low', 'apil': 'APIL', 'sqmax_ew': 'SQmax-EW', 'bsp_ws3': 'BSP-WS3', 'bsp_low_ew': 'BSP-low-EW', 'bsp_2d': 'BSP-2D', 'bsp_oir': 'BSP-OIR', 'copsq': 'COP(s,q)', 'ewa': 'EWA', 'pil': 'PIL', 'ETIS': 'ETIS', 'SCBS': 'SCBS'}
DEFAULT_RUN = Path(__file__).resolve().parents[1] / "runs/mc_final"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_root / "figures"
    compare(args.run_root, output, BASELINES, OURS, 'M',
            labels={**LABELS, CODEX_ID: LABEL}, non_sof=NON_SOF,
            formats=("png",),
            policy_colors={CODEX_ID: COLOR})
    print(output)


if __name__ == "__main__":
    main()
