#!/usr/bin/env python3
"""Plot final MC costs against the shared all-formal-policy reference."""
import argparse, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reporting.scores import load_scores
CODEX_COLOR = "#8C564B"
CODEX_LABEL = "Codex-gpt-6-astra"
from sr.core.grammar import expanded_complexity
from benchmarks.inventory_benchmark_family.mc_protocol import load_policy_set
from experiments.perishable.benchmark_campaigns.inventory_benchmark_family.mc_model import MODEL
CONTEXTS = tuple(range(2,16))
PARETO_AXIS_MAX = {'complexity': 102.0, 'parameters': 8.5}
CODEX_ID = 'codex-astra-urgency-correction'
# Baselines shown in the main comparison; Constant Order also appears in the frontier.
MAIN_BASELINES = ('base_stock', 'cbs', 'bsp_low', 'weighted_base_stock',
                  'pb_chao', 'db_chao', 'tb_zaw', CODEX_ID)
LABELS = {CODEX_ID: CODEX_LABEL, 'co':'Constant Order', 'base_stock':'Base Stock', 'cbs':'CBS',
 'bsp_low':'BSP-low',
 'weighted_base_stock':'Weighted BS', 'pb_chao':'PB-Chao', 'db_chao':'DB-Chao',
 'tb_zaw':'TB-ZAW'}
# Explicit policy colors based on the perishable publication palette.
POLICY_COLORS = {
    CODEX_ID: CODEX_COLOR,
    'co': '#E69F00',
    'base_stock': '#788898',
    'cbs': '#3979BC',
    'bsp_low': '#999933',
    'weighted_base_stock': '#117733',
    'pb_chao': '#AA4499',
    'db_chao': '#CC6677',
    'tb_zaw': '#D55E00',
}
BASELINE_LEGEND_LAYOUT = {'name_x': -.60, 'wspace': .56}


# Same annotation configuration as lead_time/plotting/mc_final.py.
# Offsets in points: (right, up), with a fixed left/center text anchor.
# Each panel is independent. Baselines use policy IDs; our policies use names.
PARETO_LABEL_OFFSETS = {
    'complexity': {
        CODEX_ID: (8, 15),
        'co': (6, 15),
        'base_stock': (6, 15),
        'cbs': (6, 15),
        'bsp_low': (-30, 13),
        'weighted_base_stock': (8, 13),
        'EABS': (-30, -15),
        'RLCS': (-30, -15),
    },
    'num_parameters': {
        CODEX_ID: (14, 28),
        'co': (6, 15),
        'base_stock': (6, 15),
        'cbs': (15, 0),
        'bsp_low': (8, 13),
        'weighted_base_stock': (-55, 13),
        'EABS': (-65, -15),
        'RLCS': (12, -15),
    },
}

def tables(root):
 roster=[p for p in json.loads((root/'POLICIES.json').read_text())['policies'] if p['id'] in LABELS or p['type']=='formula_ast']
 ids={p['id'] for p in roster}
 contexts,overall=load_scores(root,('shelf_life',))
 scores={pid:{} for pid in ids}
 for row in contexts:
  if row['policy_id'] in ids:scores[row['policy_id']][row['shelf_life']]=row['fitness']
 if any(set(values)!=set(CONTEXTS) for values in scores.values()):raise ValueError('Expected 14 shelf-life scores per plotted policy')
 aggregate={r['policy_id']:r['fitness'] for r in overall if r['policy_id'] in ids}
 if set(aggregate)!=ids:raise ValueError('Missing aggregate scores for plotted policies')
 return roster,scores,aggregate

def save(fig,out,stem):
 fig.savefig(out/(stem+'.png'),dpi=190)
 plt.close(fig)

def plot(root,out):
 roster,scores,aggregate=tables(root);out.mkdir(parents=True,exist_ok=True)
 sr=[p['id'] for p in roster if p['type']=='formula_ast'];base=[p['id'] for p in roster if p['type']!='formula_ast']
 sof=[p['id'] for p in roster if p['type']=='registered_baseline']
 definitions=load_policy_set(root/'POLICIES.json')[1];geometry={}
 for d in definitions:
  if d.id not in sr+sof:continue
  policies=[MODEL.resolve_policy(d,m,20) for m in CONTEXTS]
  geometry[d.id]={'complexity':sum(expanded_complexity(p.ast,p.motifs,allowed_state_names=[f'a{i}' for i in range(1,m+1)]) for m,p in zip(CONTEXTS,policies))/14,
   'parameters':sum(len(p.parameter_specs) for p in policies)/14}

 # Pareto CO: primitive |C(t+1)|-|C(t)|, with one rate parameter.
 # The execution cap is the action bound n; no demand-support feature is used.
 geometry['co']={'complexity':expanded_complexity({'op':'parameter','name':'rate'},{}),'parameters':1.0}
 best=min(sr,key=aggregate.get)
 # EABS: q=[A-k*I+g*E]+, the canonical three-parameter frontier policy.
 eabs=next(p for p in sr if p == 'EABS')
 selected=[eabs,best]
 labels={p:LABELS.get(p,p) for p in base}
 for i,p in enumerate(selected,1):labels[p]='EABS' if p!=best else 'RLCS'
 colors={p:POLICY_COLORS[p] for p in base}
 colors.update({p:c for p,c in zip(selected,['#244C7A','#1595A4'])})
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(12.8,4.4),sharey=True)
 for ax,key,xlabel in zip(axes,['complexity','parameters'],['Mean complexity','Mean number of parameters']):
  xkey='num_parameters' if key=='parameters' else key
  members=list(geometry);front=[];besty=math.inf
  for p in sorted(members,key=lambda p:(geometry[p][key],aggregate[p])):
   x,y=geometry[p][key],aggregate[p]
   if y<besty:front.append((x,y));besty=y
  ax.plot(*zip(*front),color='black',lw=1.35)
  for p in sr:
   xy=(geometry[p][key],aggregate[p])
   if xy in front and p not in selected and not any(math.isclose(xy[0],geometry[b][key]) and math.isclose(xy[1],aggregate[b]) for b in sof):ax.scatter(*xy,s=38,color='black')
  for i,p in enumerate(sof):
   xy=(geometry[p][key],aggregate[p]);ax.scatter(*xy,s=120,color=colors[p],marker='v' if p==CODEX_ID else '^' if p=='base_stock' else 'o',edgecolors='white',lw=.8,zorder=4)
   dx,dy=PARETO_LABEL_OFFSETS[xkey][p]
   ax.annotate('Constant Order' if p=='co' else labels[p],xy,xytext=(dx,dy),textcoords='offset points',
               ha='left',va='center',fontsize=10,color=colors[p],
               bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1})
  for i,p in enumerate(selected):
   xy=(geometry[p][key],aggregate[p]);ax.scatter(*xy,s=350,marker='*',color='#C62828',edgecolors='white',zorder=6)
   offset=PARETO_LABEL_OFFSETS[xkey][labels[p]]
   ax.annotate(labels[p]+' (our)',xy,xytext=offset,textcoords='offset points',
               fontsize=11,color='#C62828',ha='left',va='center',
               bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
  limits=PARETO_AXIS_MAX
  xmax=limits[key]*1.04;ax.set_xlim(-.03*xmax,xmax)
  ax.plot([front[-1][0],xmax],[front[-1][1]]*2,color='black',lw=1.35)
  ax.set_yscale('log');ax.set_ylim(bottom=min(aggregate.values())*.28);ax.set_xlabel(xlabel);ax.grid(alpha=.2)
 axes[0].set_ylabel('aggregate fitness');fig.subplots_adjust(left=.09,right=.99,bottom=.15,top=.97,wspace=.20)
 save(fig,out,'frontier_scores_by_M')
 shown=[p for p in base if p in MAIN_BASELINES]+selected
 legend_layout=BASELINE_LEGEND_LAYOUT
 fig=plt.figure(figsize=(18,6.5))
 grid=fig.add_gridspec(1,2,width_ratios=[1.3,1])
 # Omit only the empty interval; observations and curves cannot cross it.
 a,b=.35,14.
 for p in shown:
  values=list(scores[p].values())
  assert not any(a<v<b for v in values), 'Axis break would hide data'
  assert not (min(values)<=a and max(values)>=b), 'Curve crosses axis break'
 split=grid[0].subgridspec(2,1,height_ratios=[.28,.72],hspace=.055)
 top=fig.add_subplot(split[0]);ax=fig.add_subplot(split[1],sharex=top)
 line_axes=[top,ax]
 bar=fig.add_subplot(grid[1])
 for line_ax in line_axes:
  for p in shown:
   line_ax.plot(CONTEXTS,[scores[p][m] for m in CONTEXTS],color=colors[p],marker='v' if p==CODEX_ID else 'D' if p in selected else 'o',ms=5.5,mec='white',mew=.6,lw=2.7 if p in selected else 1.5)
 for line_ax in line_axes:
  line_ax.set_yscale('log');line_ax.grid(alpha=.2)
 ax.set_xlabel(r'$\sigma^{\mathrm{life}}$');ax.set_ylabel('configuration-level fitness');ax.grid(alpha=.2);ax.set_xticks(CONTEXTS)
 all_values=[scores[p][m] for p in shown for m in CONTEXTS]
 top.set_ylim(b,max(all_values)*1.15);ax.set_ylim(min(all_values)*.8,a)
 top.spines['bottom'].set_visible(False);ax.spines['top'].set_visible(False)
 top.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
 ax.set_ylabel('');fig.text(.018,.55,'configuration-level fitness',rotation=90,va='center',fontsize=12)
 for line_ax,y in [(top,0),(ax,1)]:
  line_ax.plot([-.012,.012],[y-.012,y+.012],transform=line_ax.transAxes,color='black',lw=1.2,clip_on=False)
 ordered=sorted(shown,key=aggregate.get);values=[aggregate[p] for p in ordered];assert min(values)>0
 lower=10**math.floor(math.log10(min(values)*.85));upper=max(values)*1.65
 bar.set_xscale('log');bar.barh(range(len(ordered)),[v-lower for v in values],left=lower,color=[colors[p] for p in ordered],height=.32)
 bar.set_yticks([]);bar.set_xlim(lower,upper);bar.set_xlabel('aggregate fitness');bar.grid(axis='x',alpha=.2);bar.set_axisbelow(True)
 for i,p in enumerate(ordered):
  group='our' if p in selected else 'SOF' if p in sof else 'non-SOF'
  bar.text(legend_layout['name_x'],i,labels[p]+f' ({group})',transform=bar.get_yaxis_transform(),ha='left',va='center',fontsize=12,color=colors[p],fontweight='bold' if p in selected else 'normal')
  bar.annotate(f'{aggregate[p]:.3g}',(aggregate[p],i),xytext=(5,0),textcoords='offset points',va='center')
 fig.subplots_adjust(left=.065,right=.98,top=.97,bottom=.13,wspace=legend_layout['wspace'])
 save(fig,out,'baseline_scores_by_M_and_aggregate_main')
 print(json.dumps([dict(id=p,label=labels[p],fitness=aggregate[p],**geometry[p]) for p in selected],indent=2))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,default=Path(__file__).resolve().parents[1]/'runs/mc_final');parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
 plot(args.run_root,args.output_dir or args.run_root/'figures')
if __name__=='__main__':main()
