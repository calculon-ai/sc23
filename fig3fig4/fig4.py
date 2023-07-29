#!/usr/bin/env python3

import argparse
import calculon
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys
import tol_colors as tc


def main(args):
  GiB = 1024 ** 3
  m80 = 80 * GiB
  m160 = 160 * GiB

  df = pandas.read_csv(args.input)
  print(f'Full data has {df.shape[0]} rows')

  # Filter the data frame into 4 subplots as follows:

  # a) 80 GiB, original optimations
  dfa = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1)) &
    (df['fused_activation'] == False) &
    (df['attention_type'] == 'multihead') &
    (df['activation_recompute'] == 'full') &
    (df['optimizer_sharding'] == False) &
    (df['tensor_par_comm_type'] == 'p2p_rs_ag') &
    (df['tensor_par_overlap'] == 'none') &
    (df['seq_par_ag_redo'] == False) &
    (df['data_par_overlap'] == False) &
    (df['proc_mem_tier1_cap_req'] <= m80)
  ]
  print(f'"a" data has {dfa.shape[0]} rows')
  assert dfa.shape[0] > 0

  # b) 80 GiB, seq_par
  dfb = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1)) &
    (df['fused_activation'] == False) &
    (df['attention_type'] == 'multihead') &
    (df['optimizer_sharding'] == False) &
    (df['tensor_par_overlap'] == 'none') &
    ((df['seq_par_ag_redo'] == True) | (df['activation_recompute'] != 'attn_only')) &
    (df['data_par_overlap'] == False) &
    (df['proc_mem_tier1_cap_req'] <= m80)
  ]
  print(f'"b" data has {dfb.shape[0]} rows')
  assert dfb.shape[0] > 0

  # c) 80 GiB, all optimizations
  dfc = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1)) &
    (df['attention_type'] == 'multihead') &
    (df['proc_mem_tier1_cap_req'] <= m80)
  ]
  print(f'"c" data has {dfc.shape[0]} rows')
  assert dfc.shape[0] > 0

  # d) 160 GiB, all optimizations
  dfd = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1)) &
    (df['attention_type'] == 'multihead') &
    (df['proc_mem_tier1_cap_req'] <= m160)
  ]
  print(f'"d" data has {dfd.shape[0]} rows')
  assert dfd.shape[0] > 0

  # Sets up the plot structure
  fig, ax = plt.subplots(2, 2, figsize=(7.5, 7))
  fig.suptitle('Megatron-1T single batch of training on 4096 A100 GPUs with '
               'various optimizations')
  vmin = 45
  vmax = 200
  tps = [1, 2, 4, 8, 16, 32]
  pps = [1, 2, 4, 8, 16, 32, 64]

  # Parses the data into 2D arrays
  for plot_idx, df in enumerate([dfa, dfb, dfc, dfd]):
    # Creates a 2D array for the raw time and mem data
    time = np.zeros((len(tps), len(pps)), dtype="float")
    mem = np.zeros((len(tps), len(pps)), dtype="float")
    for tp in tps:
      for pp in pps:
        # Filters the data frame to the TP and PP
        tp_pp = df[
          (df['tensor_par'] == tp) &
          (df['pipeline_par'] == pp)
        ]

        # Retrieves the best performing row, break ties with memory usage
        best = tp_pp[tp_pp['sample_rate'] == tp_pp['sample_rate'].max()]
        best = best[best['proc_mem_tier1_cap_req'] ==
                    best['proc_mem_tier1_cap_req'].min()]

        # Handles the result
        if best.shape[0] == 0:
          time[tps.index(tp)][pps.index(pp)] = float('inf')
          mem[tps.index(tp)][pps.index(pp)] = float('inf')
        else:
          batch_time = best.iloc[0]['total_time']
          used_mem = best.iloc[0]['proc_mem_tier1_cap_req']
          time[tps.index(tp)][pps.index(pp)] = batch_time
          mem[tps.index(tp)][pps.index(pp)] = used_mem

    # Create the plot indices
    px, py = ((0, 0), (0, 1), (1, 0), (1, 1))[plot_idx]

    # Create the title
    title = [
      '(a) Time and mem, 80 GiB\noriginal optimizations',
      '(b) Time and mem, 80 GiB\nsequence parallelism optimations',
      '(c) Time and mem, 80 GiB\nall optimizations',
      '(d) Time and mem, 160 GiB\nall optimizations'][plot_idx]

    # Format the colors based on time
    colors = copy.deepcopy(time)
    colors[np.isinf(colors)] = vmax
    im = ax[px][py].imshow(colors, cmap=tc.tol_cmap('sunset'), vmin=vmin,
                           vmax=vmax, aspect=0.75, origin='lower')

    # Show all ticks and label them with the respective list entries
    ax[px][py].set_yticks(np.arange(len(tps)))
    ax[px][py].set_xticks(np.arange(len(pps)))
    ax[px][py].set_yticklabels([f't={t}' for t in tps])
    ax[px][py].set_xticklabels([f'p={p}' for p in pps])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax[px][py].get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    for t in range(len(tps)):
      for p in range(len(pps)):
        if np.isinf(time[t, p]):
          result = '—'
        else:
          result = calculon.util.human_format(time[t, p], precision=1)
          result += '\n'
          result += calculon.util.human_format(mem[t, p], v_type='base2',
                                               precision=0)
        text = ax[px][py].text(p, t, result, ha='center', va='center',
                               color='k', size=8)

    ax[px][py].spines[:].set_visible(False)
    ax[px][py].set_xticks(np.arange(time.shape[1]+1)-.5, minor=True)
    ax[px][py].set_yticks(np.arange(mem.shape[0]+1)-.5, minor=True)
    ax[px][py].grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax[px][py].tick_params(which='minor', bottom=False, left=False)
    ax[px][py].set_title(title)


  fig.tight_layout(rect=[0, 0.03, 1, 0.98])
  fig.savefig(args.output, dpi=300, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('input', help='input file')
  ap.add_argument('output', help='output file')
  sys.exit(main(ap.parse_args()))
