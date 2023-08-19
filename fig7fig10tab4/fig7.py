#!/usr/bin/env python3

import argparse
import calculon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys
import tol_colors as tc



def main(args):
  GiB = 1024 ** 3

  # H100 infinite mem2
  print(f'Reading {args.h100_inf_input}')
  df = pandas.read_csv(args.h100_inf_input)
  print(f'Full data has {df.shape[0]} rows')
  df = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1))
  ]
  print(f'Filtered data has {df.shape[0]} rows')

  # Sets up the plot structure
  fig, ax = plt.subplots(2, 2, figsize=(7.5, 7))
  fig.suptitle('Megatron-1T training on 4096 H100 80 GiB GPUs with a\n '
               'secondary memory available for tensor offloading')

  tps = [1, 2, 4, 8, 16, 32]
  pps = [1, 2, 4, 8, 16, 32]

  # (a) Rate and mem
  rate = np.zeros((len(tps), len(pps)), dtype="float")
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
        rate[tps.index(tp)][pps.index(pp)] = 0.0
        mem[tps.index(tp)][pps.index(pp)] = float('inf')
      else:
        sample_rate = best.iloc[0]['sample_rate']
        used_mem = best.iloc[0]['proc_mem_tier1_cap_req']
        rate[tps.index(tp)][pps.index(pp)] = sample_rate
        mem[tps.index(tp)][pps.index(pp)] = used_mem

  # Format the colors based on rate
  vmin = np.min(rate[np.nonzero(rate)])
  vmax = np.max(rate)
  colors = rate.copy()
  print(colors)
  im = ax[0][0].imshow(colors, cmap=tc.tol_cmap('sunset').reversed(), vmin=vmin,
                       vmax=vmax, aspect=0.75, origin='lower')

  # Show all ticks and label them with the respective list entries
  ax[0][0].set_yticks(np.arange(len(tps)))
  ax[0][0].set_xticks(np.arange(len(pps)))
  ax[0][0].set_yticklabels([f't={t}' for t in tps])
  ax[0][0].set_xticklabels([f'p={p}' for p in pps])

  # Rotate the tick labels and set their alignment.
  plt.setp(ax[0][0].get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for t in range(len(tps)):
    for p in range(len(pps)):
      if rate[t, p] == 0.0:
        result = '—'
      else:
        result = calculon.util.human_format(rate[t, p], precision=1)
        result += '\n'
        result += calculon.util.human_format(mem[t, p], v_type='base2',
                                             precision=0)
      text = ax[0][0].text(p, t, result, ha='center', va='center',
                           color='k', size=8)

  ax[0][0].spines[:].set_visible(False)
  ax[0][0].set_xticks(np.arange(rate.shape[1]+1)-.5, minor=True)
  ax[0][0].set_yticks(np.arange(mem.shape[0]+1)-.5, minor=True)
  ax[0][0].grid(which='minor', color='w', linestyle='-', linewidth=2)
  ax[0][0].tick_params(which='minor', bottom=False, left=False)
  ax[0][0].set_title('(a) Sample rate and HBM usage')
  ax[0][0].set_ylabel('Infinite', fontsize=16)

  # (b) Full bandwidth requirements
  mem2_bw = np.zeros((len(tps), len(pps)), dtype="float")
  mem2_cap = np.zeros((len(tps), len(pps)), dtype="float")
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
        mem2_bw[tps.index(tp)][pps.index(pp)] = float('inf')
        mem2_cap[tps.index(tp)][pps.index(pp)] = float('inf')
      else:
        req_bw = best.iloc[0]['offload_mem_bw_req']
        mem2_bw[tps.index(tp)][pps.index(pp)] = req_bw
        used_mem2 = best.iloc[0]['proc_mem_tier2_cap_req']
        mem2_cap[tps.index(tp)][pps.index(pp)] = used_mem2

  # Format the colors based on capacity
  vmin = 0
  vmax = 512 * GiB
  colors = mem2_cap.copy()
  colors[np.isinf(colors)] = vmax
  im = ax[0][1].imshow(colors, cmap=tc.tol_cmap('sunset'), vmin=vmin,
                       vmax=vmax, aspect=0.75, origin='lower')

  # Show all ticks and label them with the respective list entries
  ax[0][1].set_yticks(np.arange(len(tps)))
  ax[0][1].set_xticks(np.arange(len(pps)))
  ax[0][1].set_yticklabels([f't={t}' for t in tps])
  ax[0][1].set_xticklabels([f'p={p}' for p in pps])

  # Rotate the tick labels and set their alignment.
  plt.setp(ax[0][1].get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for t in range(len(tps)):
    for p in range(len(pps)):
      if np.isinf(mem2_bw[t, p]):
        result = '—'
      else:
        result = calculon.util.human_format(mem2_bw[t, p], precision=1)
        result += '\n'
        result += calculon.util.human_format(mem2_cap[t, p], v_type='base2',
                                             precision=0)
      text = ax[0][1].text(p, t, result, ha='center', va='center',
                           color='k', size=8)

  ax[0][1].spines[:].set_visible(False)
  ax[0][1].set_xticks(np.arange(mem2_bw.shape[1]+1)-.5, minor=True)
  ax[0][1].set_yticks(np.arange(mem2_cap.shape[0]+1)-.5, minor=True)
  ax[0][1].grid(which='minor', color='w', linestyle='-', linewidth=2)
  ax[0][1].tick_params(which='minor', bottom=False, left=False)
  ax[0][1].set_title('(b) Offloading bandwidth and usage')

  # H100 real mem2
  del df
  print(f'Reading {args.h100_real_input}')
  df = pandas.read_csv(args.h100_real_input)
  print(f'Full data has {df.shape[0]} rows')
  df = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1))
  ]
  print(f'Filtered data has {df.shape[0]} rows')

  # (c) Rate and mem
  rate = np.zeros((len(tps), len(pps)), dtype="float")
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
        rate[tps.index(tp)][pps.index(pp)] = 0.0
        mem[tps.index(tp)][pps.index(pp)] = float('inf')
      else:
        sample_rate = best.iloc[0]['sample_rate']
        used_mem = best.iloc[0]['proc_mem_tier1_cap_req']
        rate[tps.index(tp)][pps.index(pp)] = sample_rate
        mem[tps.index(tp)][pps.index(pp)] = used_mem

  # Format the colors based on rate
  vmin = np.min(rate[np.nonzero(rate)])
  vmax = np.max(rate)
  colors = rate.copy()
  im = ax[1][0].imshow(colors, cmap=tc.tol_cmap('sunset').reversed(), vmin=vmin,
                       vmax=vmax, aspect=0.75, origin='lower')

  # Show all ticks and label them with the respective list entries
  ax[1][0].set_yticks(np.arange(len(tps)))
  ax[1][0].set_xticks(np.arange(len(pps)))
  ax[1][0].set_yticklabels([f't={t}' for t in tps])
  ax[1][0].set_xticklabels([f'p={p}' for p in pps])

  # Rotate the tick labels and set their alignment.
  plt.setp(ax[1][0].get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for t in range(len(tps)):
    for p in range(len(pps)):
      if rate[t, p] == 0:
        result = '—'
      else:
        result = calculon.util.human_format(rate[t, p], precision=1)
        result += '\n'
        result += calculon.util.human_format(mem[t, p], v_type='base2',
                                             precision=0)
      text = ax[1][0].text(p, t, result, ha='center', va='center',
                           color='k', size=8)

  ax[1][0].spines[:].set_visible(False)
  ax[1][0].set_xticks(np.arange(rate.shape[1]+1)-.5, minor=True)
  ax[1][0].set_yticks(np.arange(mem.shape[0]+1)-.5, minor=True)
  ax[1][0].grid(which='minor', color='w', linestyle='-', linewidth=2)
  ax[1][0].tick_params(which='minor', bottom=False, left=False)
  ax[1][0].set_title('(c) Sample rate and HBM usage')
  ax[1][0].set_ylabel('512 GiB @ 100 GB/s', fontsize=16)

  # (d) Full bandwidth requirements
  mem2_bw = np.zeros((len(tps), len(pps)), dtype="float")
  mem2_cap = np.zeros((len(tps), len(pps)), dtype="float")
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
        mem2_bw[tps.index(tp)][pps.index(pp)] = float('inf')
        mem2_cap[tps.index(tp)][pps.index(pp)] = float('inf')
      else:
        req_bw = best.iloc[0]['offload_mem_bw_req']
        mem2_bw[tps.index(tp)][pps.index(pp)] = min(100e9, req_bw)
        used_mem2 = best.iloc[0]['proc_mem_tier2_cap_req']
        mem2_cap[tps.index(tp)][pps.index(pp)] = used_mem2

  # Format the colors based on capacity
  vmin = 0
  vmax = 512 * GiB
  colors = mem2_cap.copy()
  colors[np.isinf(colors)] = vmax
  im = ax[1][1].imshow(colors, cmap=tc.tol_cmap('sunset'), vmin=vmin,
                       vmax=vmax, aspect=0.75, origin='lower')

  # Show all ticks and label them with the respective list entries
  ax[1][1].set_yticks(np.arange(len(tps)))
  ax[1][1].set_xticks(np.arange(len(pps)))
  ax[1][1].set_yticklabels([f't={t}' for t in tps])
  ax[1][1].set_xticklabels([f'p={p}' for p in pps])

  # Rotate the tick labels and set their alignment.
  plt.setp(ax[1][1].get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for t in range(len(tps)):
    for p in range(len(pps)):
      if np.isinf(mem2_bw[t, p]):
        result = '—'
      else:
        result = calculon.util.human_format(mem2_bw[t, p], precision=1)
        result += '\n'
        result += calculon.util.human_format(mem2_cap[t, p], v_type='base2',
                                             precision=0)
      text = ax[1][1].text(p, t, result, ha='center', va='center',
                           color='k', size=8)

  ax[1][1].spines[:].set_visible(False)
  ax[1][1].set_xticks(np.arange(mem2_bw.shape[1]+1)-.5, minor=True)
  ax[1][1].set_yticks(np.arange(mem2_cap.shape[0]+1)-.5, minor=True)
  ax[1][1].grid(which='minor', color='w', linestyle='-', linewidth=2)
  ax[1][1].tick_params(which='minor', bottom=False, left=False)
  ax[1][1].set_title('(d) Offloading bandwidth and usage')


  fig.tight_layout(rect=[0, 0.03, 1, 0.98])
  fig.savefig(args.output, dpi=300, transparent=False, bbox_inches='tight')


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('h100_inf_input', help='H100 infinite mem2 input file')
  ap.add_argument('h100_real_input', help='H100 real mem2 input file')
  ap.add_argument('output', help='output file')
  sys.exit(main(ap.parse_args()))
