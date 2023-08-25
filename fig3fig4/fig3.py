#!/usr/bin/env python3

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import sys


def main(args):
  dfa = pandas.read_csv(args.input)
  print(f'Full data has {dfa.shape[0]} rows')

  # Filter the data frame
  df = dfa[
    (dfa['tensor_par_net'] == 0) &
    ((dfa['pipeline_par_net'] == 1) | (dfa['pipeline_par'] == 1)) &
    ((dfa['data_par_net'] == 1) | (dfa['data_par'] == 1)) &
    (dfa['fused_activation'] == False) &
    (dfa['attention_type'] == 'multihead') &
    (dfa['seq_par_ag_redo'] == False) &
    (dfa['tensor_par_overlap'] == 'none') &
    (dfa['data_par_overlap'] == False) &
    (dfa['optimizer_sharding'] == True) &
    (dfa['activation_recompute'] == 'attn_only') &
    (dfa['tensor_par_comm_type'] == 'rs_ag') &
    (dfa['microbatch_size'] == 1)
  ]
  print(f'Filtered data has {df.shape[0]} rows')
  assert df.shape[0] > 0

  df.to_csv(args.input.replace('all', 'filtered'))  ##############################DEBUG

  # TP vs PP, DP fixed to 32
  tps = [1, 2, 4, 8, 16, 32]
  dp32 = df[df['data_par'] == 32]
  dp32_bests = {}
  for tp in tps:
    tf = dp32[dp32['tensor_par'] == tp]
    assert tf.shape[0] > 0, f'dp=32 tp={tp} has 0 results'
    t = tf[tf['sample_rate'] == tf['sample_rate'].max()]
    dp32_bests[tp] = t

  # PP vs DP, TP fixed to 8
  pps = [1, 2, 4, 8, 16, 32, 64, 128]
  tp8 = df[df['tensor_par'] == 8]
  tp8_bests = {}
  for pp in pps:
    pf = tp8[tp8['pipeline_par'] == pp]
    assert pf.shape[0] > 0, f'tp=8 pp={pp} has 0 results'
    t = pf[pf['sample_rate'] == pf['sample_rate'].max()]
    tp8_bests[pp] = t

  # TP vs DP, PP fixed to 32
  pp32 = df[df['pipeline_par'] == 32]
  pp32_bests = {}
  for tp in tps:
    tf = pp32[pp32['tensor_par'] == tp]
    assert tf.shape[0] > 0, f'pp=32 tp={tp} has 0 results'
    t = tf[tf['sample_rate'] == tf['sample_rate'].max()]
    pp32_bests[tp] = t

  # 2x3 subplots
  fig, ax = plt.subplots(2, 3, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 1.6, 1]})
  fig.suptitle('Megatron-1T single batch training on 4096 A100 GPUs '
               'with various parallelism strategies', fontsize=18, y=0.97)
  bar_width = 0.5
  time_min = 0
  time_max = 100
  mem_min = 0
  mem_max = 300

  time_colors = [
    '#AA4499',
    '#882255',
    '#CC6677',
    '#DDCC77',
    '#999933',
    '#117733',
    '#44AA99',
    '#88CCEE'
  ]

  mem_colors = [
    '#EE6677',
    '#AA3377',
    '#228833',
    '#66CCEE',
    '#4477AA',
  ]

  for idx, ds, keys, title, labels in [
      (0, dp32_bests, tps, 'TP vs PP (DP=32)',
       [f't={t}\np={4096//32//t}' for t in tps]),
      (1, tp8_bests, pps, 'PP vs DP (TP=8)',
       [f'p={p}\nd={4096//8//p}' for p in pps]),
      (2, pp32_bests, tps, 'TP vs DP (PP=32)',
       [f't={t}\nd={4096//32//t}' for t in tps])]:

    # Time plot (top row)
    fw_time = [ds[t].iloc[0]['fw_time'] for t in keys]
    bw_time = [ds[t].iloc[0]['bw_time'] for t in keys]
    optim_time = [ds[t].iloc[0]['optim_step_time'] for t in keys]
    recomp_time = [ds[t].iloc[0]['recompute_time'] for t in keys]
    bubble_time = [ds[t].iloc[0]['bubble_time'] for t in keys]
    tp_time = [ds[t].iloc[0]['tp_comm_exposed_time'] for t in keys]
    pp_time = [ds[t].iloc[0]['pp_comm_exposed_time'] for t in keys]
    dp_time = [ds[t].iloc[0]['dp_comm_exposed_time'] for t in keys]

    ax[0][idx].bar(labels, fw_time, bar_width,
                   label='FW pass', color=time_colors[0])
    ax[0][idx].bar(labels, bw_time, bar_width, bottom=fw_time,
                   label='BW pass', color=time_colors[1])
    ax[0][idx].bar(labels, optim_time, bar_width,
                   bottom=[sum(x) for x in zip(fw_time, bw_time)],
                   label='Optim step', color=time_colors[2])
    ax[0][idx].bar(labels, bubble_time, bar_width,
                   bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time)],
                   label='PP bubble', color=time_colors[3])
    ax[0][idx].bar(labels, recomp_time, bar_width,
                   bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time,
                                               bubble_time)],
                   label='FW recompute', color=time_colors[4])
    ax[0][idx].bar(labels, tp_time, bar_width,
                   bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time,
                                               bubble_time, recomp_time)],
                   label='TP comm', color=time_colors[5])
    ax[0][idx].bar(labels, pp_time, bar_width,
                   bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time,
                                               bubble_time, recomp_time,
                                               tp_time)],
                   label='PP comm', color=time_colors[6])
    ax[0][idx].bar(labels, dp_time, bar_width,
                   bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time,
                                               bubble_time, recomp_time,
                                               tp_time, pp_time)],
                   label='DP comm', color=time_colors[7])

    plt.setp(ax[0][idx].get_xticklabels(), fontsize=11)
    plt.setp(ax[0][idx].get_yticklabels(), fontsize=11)
    ax[0][idx].set_ylabel('Time, s', fontsize=12)
    ax[0][idx].set_ylim([time_min, time_max])
    ax[0][idx].set_title(f'{title} batch time', fontsize=12)

    # Mem plot
    weight = [ds[t].iloc[0]['weight_space']/1024**3 for t in keys]
    act_space = [ds[t].iloc[0]['act_space']/1024**3 for t in keys]
    weight_grad = [ds[t].iloc[0]['act_grad_space']/1024**3 for t in keys]
    act_grad = [ds[t].iloc[0]['weight_grad_space']/1024**3 for t in keys]
    optim_space = [ds[t].iloc[0]['optimizer_space']/1024**3 for t in keys]

    ax[1][idx].bar(labels, weight, bar_width, label='Weight',
                   color=mem_colors[0])
    ax[1][idx].bar(labels, act_space, bar_width, bottom=weight,
                 label='Activation', color=mem_colors[1])
    ax[1][idx].bar(labels, weight_grad, bar_width,
                 bottom=[sum(x) for x in zip(weight, act_space)],
                 label='Weight\ngradients', color=mem_colors[2])
    ax[1][idx].bar(labels, act_grad, bar_width,
                 bottom=[sum(x) for x in zip(weight, act_space, weight_grad)],
                 label='Activation\ngradients', color=mem_colors[3])
    ax[1][idx].bar(labels, optim_space, bar_width,
                 bottom=[sum(x) for x in zip(weight, act_space, weight_grad,
                                             act_grad)],
                 label='Optimizer space', color=mem_colors[4])

    plt.setp(ax[1][idx].get_xticklabels(), fontsize=11)
    plt.setp(ax[1][idx].get_yticklabels(), fontsize=11)
    ax[1][idx].set_ylabel('Size, GB', fontsize=12)
    ax[1][idx].set_ylim([mem_min, mem_max])
    ax[1][idx].set_title(f'{title} memory consumption', fontsize=12)

  # Time legend
  ax[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
                  fancybox=True, shadow=True, ncol=2, fontsize=12)

  # Memory consumption legend
  ax[1][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
                  fancybox=True, shadow=True, ncol=2, fontsize=12)

  # Create plotfile
  fig.tight_layout(rect=[0, 0.01, 1, 0.99])
  fig.savefig(args.output, dpi=300, transparent=False)



if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('input', help='input file')
  ap.add_argument('output', help='output file')
  sys.exit(main(ap.parse_args()))
