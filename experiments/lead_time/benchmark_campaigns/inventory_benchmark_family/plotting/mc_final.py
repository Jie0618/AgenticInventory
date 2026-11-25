#!/usr/bin/env python3
"""Plot the final frontier and baseline MC scores for L=2–15."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reporting.scores import load_scores
CODEX_COLOR = "#8C564B"
CODEX_LABEL = "Codex-gpt-6-astra"

LEAD_TIMES = tuple(range(2, 16))
CODEX_ID = 'codex-astra-capped-survival'
SOF_IDS = {'co', 'base_stock', 'cbs', 'tilted_pic',
           'weighted_tilted_pic', 'aips_hybrid', CODEX_ID}
LABELS = {
    'co': 'Constant Order', 'base_stock': 'Base stock', 'cbs': 'CBS',
    'tilted_pic': 'Tilted PIC', 'weighted_tilted_pic': 'Weighted Tilted PIC',
    'myopic2': 'Myopic-2', 'pil': 'PIL',
    'aips_hybrid': 'AIPS Hybrid', 'llm_or_projected_mv_capped': 'LLM-OR Projected MV',
    CODEX_ID: CODEX_LABEL,
}
IMBALANCED_ID = 'IA-SCDS'
# Separated hues distinguish adjacent baseline markers. Keep policy
# colors identical across the Pareto and main comparison.
POLICY_COLORS = {
    CODEX_ID: CODEX_COLOR,
    'co': '#D55E00', 'base_stock': '#788898',
    'cbs': '#0072B2',
    'tilted_pic': '#AA4499', 'weighted_tilted_pic': '#7A6A20',
    'myopic2': '#56B4E9', 'pil': '#7C5AA6', 'aips_hybrid': '#E69F00',
    'llm_or_projected_mv_capped': '#44AA99',
}
# Restore the previous green accent in the Pareto panels only.
PARETO_COLORS = {**POLICY_COLORS, 'aips_hybrid': '#009E73'}
FRONTIER_POINT_AREA = 38
BASELINE_POINT_AREA = 150
SELECTED_POINT_AREA = 400
FRONTIER_LINEWIDTH = 1.35
# Distinct footprints preserve both identities at coincident coordinates.
# CO is a filled circle; a smaller base-stock triangle remains visible over it.
PARETO_BASELINE_MARKERS = {
    CODEX_ID: {'marker': 'v', 's': BASELINE_POINT_AREA},
    'aips_hybrid': {'marker': 'p', 's': BASELINE_POINT_AREA},
    'co': {'marker': 'o', 's': 120},
    'base_stock': {'marker': '^', 's': 100},
    'cbs': {'marker': 'D', 's': BASELINE_POINT_AREA},
    'tilted_pic': {'marker': 'X', 's': BASELINE_POINT_AREA},
    'weighted_tilted_pic': {'marker': 'h', 's': BASELINE_POINT_AREA},
}



# Point-label offsets in points: (right, up), relative to a fixed left/center
# text anchor. Signs never change alignment, so movement is continuous at zero.
# Each panel is independent.
PARETO_LABEL_OFFSETS = {
    'complexity': {
        CODEX_ID: (9, 15),
        'aips_hybrid': (10, 1),
        'co': (9, 12), 'base_stock': (9, -2),
        'cbs': (9, 10),
        'tilted_pic': (12, -1), 'weighted_tilted_pic': (-100, -20),
        'SCDS': (-27, -20),
        'IA-SCDS': (-30, -20),
    },
    'num_parameters': {
        CODEX_ID: (9, 15),
        'aips_hybrid': (-70, 8),
        'co': (9, 12), 'base_stock': (9, -18),
        'cbs': (9, 10),
        'tilted_pic': (12, -1), 'weighted_tilted_pic': (-110, -12),
        'SCDS': (11, 7),
        'IA-SCDS': (11, -12),
    },
}
BASELINE_LEGEND_LAYOUT = {'name_x': -.60, 'wspace': .56}

def score_tables(root):
    """Read saved scores for the listed baselines and symbolic policies."""
    roster = [p for p in json.loads((root / 'POLICIES.json').read_text())['policies']
              if p['id'] in LABELS or p['type'] == 'formula_ast']
    ids = {p['id'] for p in roster}
    contexts, overall = load_scores(root, ('lead_time',))
    scores = {pid: {} for pid in ids}
    for row in contexts:
        if row['policy_id'] in ids:
            scores[row['policy_id']][row['lead_time']] = row['fitness']
    if any(set(values) != set(LEAD_TIMES) for values in scores.values()):
        raise ValueError('Expected 14 lead-time scores per plotted policy')
    aggregate = {row['policy_id']: row for row in overall if row['policy_id'] in ids}
    if set(aggregate) != ids:
        raise ValueError('Missing aggregate scores for plotted policies')
    return roster, scores, aggregate


def _style_axis(ax):
    ax.grid(axis='y', color='#dce2e9', linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_xticks(LEAD_TIMES)
    ax.set_xlabel(r'$\sigma^{\mathrm{lead}}$')
    ax.set_ylabel('configuration-level fitness')
    # Includes exact-zero scores without adding an arbitrary epsilon.
    ax.set_yscale('symlog', linthresh=.1, linscale=.5)
    ax.set_ylim(bottom=0)


def plot(root, output):
    roster, scores, all_aggregate = score_tables(root)
    if IMBALANCED_ID not in scores:
        raise ValueError('Final leadtime figures require IA-SCDS measurements')
    output.mkdir(parents=True, exist_ok=True)
    sr = [p for p in roster if p['type'] == 'formula_ast']
    baselines = [p['id'] for p in roster if p['type'] != 'formula_ast']
    labels = {pid: LABELS.get(pid, pid) for pid in scores}
    strict = next(p['id'] for p in sr if p.get('role') == 'strict_scds')
    labels[strict] = 'SCDS'
    if IMBALANCED_ID in scores:
        labels[IMBALANCED_ID] = 'IA-SCDS'
    highlighted = [strict] + ([IMBALANCED_ID] if IMBALANCED_ID in scores else [])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'savefig.facecolor': 'white'})
    from benchmarks.inventory_benchmark_family.mc_protocol import load_policy_set
    from experiments.lead_time.benchmark_campaigns.inventory_benchmark_family.mc_model import MODEL
    from sr.core.grammar import expanded_complexity
    definitions = load_policy_set(root / 'POLICIES.json')[1]
    geometry = {}
    for definition in definitions:
        if definition.id not in scores or (definition.id not in SOF_IDS and definition.type != 'formula_ast'):
            continue
        complexities, counts = [], []
        for L in LEAD_TIMES:
            policy = MODEL.resolve_policy(definition, L, 20)
            complexities.append(expanded_complexity(policy.ast, policy.motifs,
                allowed_state_names=['x', *(f'p{i}' for i in range(1, L+1))]))
            counts.append(len(policy.parameter_specs))
        geometry[definition.id] = {'complexity': sum(complexities)/len(complexities),
                                   'num_parameters': sum(counts)/len(counts)}
    # CO is the primitive cumulative integer increment |C(t+1)|-|C(t)|.
    # The execution cap is the action bound n, independent of demand support.
    # Count the rate once, as for the primitive constant-order q=r family.
    geometry['co'] = {
        'complexity': expanded_complexity({'op': 'parameter', 'name': 'rate'}, {}),
        'num_parameters': 1.0,
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), sharey=True)
    black_markers = []
    for ax, frontier, xkey, title in zip(axes, ('frontier', 'parameter_frontier'),
            ('complexity', 'num_parameters'),
            ('Expression-complexity frontier', 'Parameter-count frontier')):
        members = list(geometry)
        envelope = []
        best_score = math.inf
        for pid in sorted(members, key=lambda pid: (geometry[pid][xkey], all_aggregate[pid]['fitness'])):
            x, y = geometry[pid][xkey], all_aggregate[pid]['fitness']
            if y < best_score:
                envelope.append((x, y))
                best_score = y
        assert all(x2 > x1 and y2 < y1 for (x1, y1), (x2, y2) in zip(envelope, envelope[1:]))
        for index, pid in enumerate(members):
            if pid in highlighted or pid in SOF_IDS:
                continue
            x, y = geometry[pid][xkey], all_aggregate[pid]['fitness']
            if (x, y) in envelope:
                marker = ax.scatter(x, y, s=FRONTIER_POINT_AREA, color='black', edgecolors='black', linewidth=.5, zorder=3)
                black_markers.append((ax, pid, xkey, marker))
        ax.plot([x for x, _ in envelope], [y for _, y in envelope],
                color='black', linewidth=FRONTIER_LINEWIDTH, zorder=2, label='OUR-SR Pareto frontier')
        for i, pid in enumerate(pid for pid in baselines if pid in SOF_IDS):
            x, y = geometry[pid][xkey], all_aggregate[pid]['fitness']
            style = PARETO_BASELINE_MARKERS[pid]
            hollow = style.get('hollow', False)
            ax.scatter(x, y, s=style['s'], marker=style['marker'],
                       facecolors='none' if hollow else PARETO_COLORS[pid],
                       edgecolors=PARETO_COLORS[pid] if hollow else 'white',
                       linewidth=1.8 if hollow else .8, zorder=4, label=labels[pid])
            dx, dy = PARETO_LABEL_OFFSETS[xkey][pid]
            ax.annotate('Constant Order' if pid == 'co' else labels[pid], (x, y), xytext=(dx, dy), textcoords='offset points',
                        fontsize=11, ha='left', va='center',
                        color=PARETO_COLORS[pid],
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': 1.3})
        for pid in highlighted:
            x, y = geometry[pid][xkey], all_aggregate[pid]['fitness']
            color = '#C62828'
            ax.scatter(x, y, s=SELECTED_POINT_AREA,
                       marker='*', color=color,
                       edgecolors='white', linewidth=1.2, zorder=6)
            offset = PARETO_LABEL_OFFSETS[xkey][labels[pid]]
            ax.annotate(f'{labels[pid]} (our)', (x, y), xytext=offset, textcoords='offset points',
                        fontsize=12, color=color, ha='left', va='center',
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1.2})
        ax.set_yscale('log')
        ax.set_ylim(bottom=.085)
        ax.set_xlabel('Mean complexity' if xkey == 'complexity' else 'Mean number of parameters')
        if xkey == 'num_parameters':
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(13)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        xmax = max([x for x, _ in envelope] + [geometry[pid][xkey] for pid in baselines if pid in SOF_IDS])
        right_edge = xmax * 1.035
        ax.set_xlim(left=-.045 * right_edge, right=right_edge)
        ax.plot([envelope[-1][0], right_edge], [envelope[-1][1]] * 2,
                color='black', linewidth=FRONTIER_LINEWIDTH, zorder=2)
    axes[0].set_ylabel('aggregate fitness', fontsize=13)
    fig.subplots_adjust(left=.09, right=.99, top=.97, bottom=.15, wspace=.20)
    # Resolve visual overlap after the shared scale and final layout are fixed.
    fig.canvas.draw()
    for ax, pid, xkey, marker in black_markers:
        xy = ax.transData.transform((geometry[pid][xkey], all_aggregate[pid]['fitness']))
        overlaps = any(math.dist(xy, ax.transData.transform(
            (geometry[bid][xkey], all_aggregate[bid]['fitness'])))
            <= (math.sqrt(FRONTIER_POINT_AREA) + math.sqrt(PARETO_BASELINE_MARKERS[bid]['s'])) / 2 * fig.dpi / 72
            for bid in baselines if bid in SOF_IDS)
        marker.set_visible(not overlaps)
    fig.savefig(output / 'frontier_scores_by_L.png', dpi=190)
    plt.close(fig)
    # Show the two selected SCDS policies.
    selected_our = [pid for pid in highlighted if pid != strict]
    comparison = baselines + [strict] + selected_our
    comparison_labels = {pid: f"{labels[pid]} ({'SOF' if pid in SOF_IDS else 'non-SOF'})"
                         for pid in baselines}
    comparison_labels[strict] = 'SCDS (our)'
    if IMBALANCED_ID in scores:
        comparison_labels[IMBALANCED_ID] = 'IA-SCDS (our)'
    aggregate = all_aggregate
    bcolors = {**POLICY_COLORS, strict: '#0C6570', IMBALANCED_ID: '#332288'}
    legend_layout = BASELINE_LEGEND_LAYOUT
    shown = comparison
    fig, (ax, bar) = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={'width_ratios': [1.3, 1]})
    for pid in shown:
        ax.plot(LEAD_TIMES, [scores[pid][L] for L in LEAD_TIMES], color=bcolors[pid],
                marker='v' if pid == CODEX_ID else 'P' if pid == IMBALANCED_ID else ('D' if pid == strict else ('s' if pid in selected_our else 'o')),
                markersize=5.5, markeredgecolor='white', markeredgewidth=.6,
                linewidth=2.7 if pid in selected_our or pid == strict else 1.5,
                linestyle='-',
                label=comparison_labels[pid])
    _style_axis(ax)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    visible_values = [scores[pid][L] for pid in shown for L in LEAD_TIMES]
    if min(visible_values) > 0:
        ax.set_yscale('log')
        ax.set_ylim(min(visible_values) / 1.18, max(visible_values) * 1.18)
    ordered = sorted(shown, key=lambda pid: aggregate[pid]['fitness'])
    values = [aggregate[pid]['fitness'] for pid in ordered]
    if min(values) <= 0:
        raise ValueError('Log aggregate bars require positive fitness')
    lower = 10 ** math.floor(math.log10(min(values) * .85))
    upper = 10 ** math.ceil(math.log10(max(values) * 1.15))
    bar.set_xscale('log')
    bar.barh(range(len(ordered)), [value - lower for value in values], left=lower,
             color=[bcolors[pid] for pid in ordered], height=.32)
    bar.set_yticks(range(len(ordered)), [])
    bar.tick_params(axis='y', length=0)
    # One left-aligned name column in the gap, with rows shared by bars.
    for row, pid in enumerate(ordered):
        bar.text(legend_layout['name_x'], row,
                 comparison_labels[pid],
                 transform=bar.get_yaxis_transform(), ha='left', va='center',
                 fontsize=13, color=bcolors[pid], clip_on=False,
                 fontweight='bold' if pid in selected_our or pid == strict else 'normal')
    bar.set_xlabel('aggregate fitness', fontsize=14)
    bar.tick_params(axis='x', which='major', labelsize=12)
    bar.grid(axis='x', alpha=.2)
    bar.set_axisbelow(True)
    bar.spines[['top', 'right']].set_visible(False)
    bar.set_xlim(lower, upper)
    for i, value in enumerate(values):
        bar.annotate(f'{value:.3g}', (value, i), xytext=(5, 0),
                     textcoords='offset points', va='center', fontsize=12)
    fig.subplots_adjust(left=.065, right=.98, top=.97, bottom=.13, wspace=legend_layout['wspace'])
    fig.savefig(output / 'baseline_scores_by_L_and_aggregate_main.png', dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=REPOSITORY_ROOT / 'experiments/lead_time/benchmark_campaigns/inventory_benchmark_family/runs/mc_final')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    plot(args.run_root.resolve(), (args.output_dir or args.run_root/'figures').resolve())


if __name__ == '__main__':
    main()
