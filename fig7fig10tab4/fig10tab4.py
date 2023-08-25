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
  df = pandas.read_csv(args.input)
  print(f'Full data has {df.shape[0]} rows')

  # Baseline
  d1 = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1)) &
    (df['fused_activation'] == False) &
    (df['attention_type'] == 'multihead') &
    (df['seq_par_ag_redo'] == False) &
    (df['tensor_par_overlap'] == 'none') &
    (df['data_par_overlap'] == False) &
    (df['optimizer_sharding'] == False) &
    (df['activation_recompute'] == 'full') &
    ((df['tensor_par_comm_type'] == 'ar') |
     (df['tensor_par_comm_type'] == 'p2p_rs_ag')) &
    (df['tensor_par'] == 8) &
    (df['pipeline_par'] == 64) &
    (df['data_par'] == 8) &
    (df['microbatch_size'] == 1) &
    (df['pipeline_interleaving'] == 2) &
    (df['weight_offload'] == False) &
    (df['activations_offload'] == False) &
    (df['optimizer_offload'] == False)
  ]
  print(f'Baseline data has {d1.shape[0]} rows')
  assert d1.shape[0] > 0
  e1 = d1[d1['sample_rate'] == d1['sample_rate'].max()]

  # Seq Par
  d2 = df[
    (df['tensor_par_net'] == 0) &
    ((df['pipeline_par_net'] == 1) | (df['pipeline_par'] == 1)) &
    ((df['data_par_net'] == 1) | (df['data_par'] == 1)) &
    (df['fused_activation'] == False) &
    (df['attention_type'] == 'multihead') &
    (df['seq_par_ag_redo'] == True) &
    (df['tensor_par_overlap'] == 'none') &
    (df['data_par_overlap'] == False) &
    (df['optimizer_sharding'] == False) &
    (df['activation_recompute'] == 'attn_only') &
    (df['tensor_par'] == 8) &
    (df['pipeline_par'] == 64) &
    (df['data_par'] == 8) &
    (df['microbatch_size'] == 1) &
    (df['weight_offload'] == False) &
    (df['activations_offload'] == False) &
    (df['optimizer_offload'] == False)
  ]
  print(f'Seq par data has {d2.shape[0]} rows')
  assert d2.shape[0] > 0
  e2 = d2[d2['sample_rate'] == d2['sample_rate'].max()]

  # All software optimizations (no offloading)
  d3 = df[
    (df['weight_offload'] == False) &
    (df['activations_offload'] == False) &
    (df['optimizer_offload'] == False)
  ]
  print(f'SwOpt data has {d3.shape[0]} rows')
  assert d3.shape[0] > 0
  e3 = d3[d3['sample_rate'] == d3['sample_rate'].max()]

  # All software optimizations with offloading
  d4 = df
  print(f'Offloading data has {d4.shape[0]} rows')
  assert d4.shape[0] > 0
  e4 = d4[d4['sample_rate'] == d4['sample_rate'].max()]


  # Figure 10
  fig, ax = plt.subplots(1, 2, figsize=(8, 4))
  fig.suptitle("Calculon results compared to State-of-the-Art",
               fontsize=18, y=1.01)
  width = 0.5
  tmin=0
  tmax=26
  mmin=0
  mmax=80
  colors = [
    '#AA4499',
    '#882255',
    '#CC6677',
    '#DDCC77',
    '#999933',
    '#117733',
    '#44AA99',
    '#88CCEE'
  ]
  labels = [
    "Baseline\nrecompute",
    "SOTA\nseq par",
    "Calculon\nSW algos",
    "Calculon\nSW algos+\noffload"
  ]
  fw_time = [e1.iloc[0]['fw_time'],
             e2.iloc[0]['fw_time'],
             e3.iloc[0]['fw_time'],
             e4.iloc[0]['fw_time']]
  bw_time = [e1.iloc[0]['bw_time'],
             e2.iloc[0]['bw_time'],
             e3.iloc[0]['bw_time'],
             e4.iloc[0]['bw_time']]
  optim_time = [e1.iloc[0]['optim_step_time'],
                e2.iloc[0]['optim_step_time'],
                e3.iloc[0]['optim_step_time'],
                e4.iloc[0]['optim_step_time']]
  recomp_time = [e1.iloc[0]['recompute_time'],
                 e2.iloc[0]['recompute_time'],
                 e3.iloc[0]['recompute_time'],
                 e4.iloc[0]['recompute_time']]
  bubble_time = [e1.iloc[0]['bubble_time'],
                 e2.iloc[0]['bubble_time'],
                 e3.iloc[0]['bubble_time'],
                 e4.iloc[0]['bubble_time']]
  tp_time = [e1.iloc[0]['tp_comm_exposed_time']
             + e1.iloc[0]['recomm_exposed_time'],
             e2.iloc[0]['tp_comm_exposed_time']
             + e2.iloc[0]['recomm_exposed_time'],
             e3.iloc[0]['tp_comm_exposed_time']
             + e3.iloc[0]['recomm_exposed_time'],
             e4.iloc[0]['tp_comm_exposed_time']
             + e4.iloc[0]['recomm_exposed_time']]
  pp_time = [e1.iloc[0]['pp_comm_exposed_time'],
             e2.iloc[0]['pp_comm_exposed_time'],
             e3.iloc[0]['pp_comm_exposed_time'],
             e4.iloc[0]['pp_comm_exposed_time']]
  dp_time = [e1.iloc[0]['dp_comm_exposed_time'],
             e2.iloc[0]['dp_comm_exposed_time'],
             e3.iloc[0]['dp_comm_exposed_time'],
             e4.iloc[0]['dp_comm_exposed_time']]

  ax[0].bar(labels, fw_time, width,
            label='FW pass', color=colors[0])
  ax[0].bar(labels, bw_time, width, bottom=fw_time,
            label='BW pass', color=colors[1])
  ax[0].bar(labels, optim_time, width,
            bottom=[sum(x) for x in zip(fw_time, bw_time)],
            label='Optim step', color=colors[2])
  ax[0].bar(labels, bubble_time, width,
            bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time)],
            label='PP bubble', color=colors[3])
  ax[0].bar(labels, recomp_time, width,
            bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time, bubble_time)],
            label='FW recompute', color=colors[4])
  ax[0].bar(labels, tp_time, width,
            bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time, bubble_time, recomp_time)],
            label='TP comm', color=colors[5])
  ax[0].bar(labels, pp_time, width,
            bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time, bubble_time, recomp_time, tp_time)],
            label='PP comm', color=colors[6])
  ax[0].bar(labels, dp_time, width,
            bottom=[sum(x) for x in zip(fw_time, bw_time, optim_time, bubble_time, recomp_time, tp_time, pp_time)],
            label='DP comm', color=colors[7])

  ax[0].set_ylabel('Time, s', fontsize=12)
  ax[0].set_ylim([tmin, tmax])
  plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor", fontsize=11)
  ax[0].legend(loc='lower center', bbox_to_anchor=(0.37, -0.65),
               fancybox=True, shadow=True, ncol=2, fontsize=10)
  ax[0].set_title('Batch time', fontsize=12)

  weight = [e1.iloc[0]['weight_space']/1024**3,
            e2.iloc[0]['weight_space']/1024**3,
            e3.iloc[0]['weight_space']/1024**3,
            e4.iloc[0]['weight_space_with_offload']/1024**3]
  act_space = [e1.iloc[0]['act_space']/1024**3
               + e1.iloc[0]['act_checkpoint_size']/1024**3,
               e2.iloc[0]['act_space']/1024**3
               + e2.iloc[0]['act_checkpoint_size']/1024**3,
               e3.iloc[0]['act_space']/1024**3
               + e3.iloc[0]['act_checkpoint_size']/1024**3,
               e4.iloc[0]['act_space_with_offload']/1024**3
               + e4.iloc[0]['act_checkpoint_size_with_offload']/1024**3]
  weight_grad = [e1.iloc[0]['weight_grad_space']/1024**3,
                 e2.iloc[0]['weight_grad_space']/1024**3,
                 e3.iloc[0]['weight_grad_space']/1024**3,
                 e4.iloc[0]['weight_grad_space_with_offload']/1024**3]
  act_grad = [e1.iloc[0]['act_grad_space']/1024**3,
              e2.iloc[0]['act_grad_space']/1024**3,
              e3.iloc[0]['act_grad_space']/1024**3,
              e4.iloc[0]['act_grad_space_with_offload']/1024**3]
  optim_space = [e1.iloc[0]['optimizer_space']/1024**3,
                 e2.iloc[0]['optimizer_space']/1024**3,
                 e3.iloc[0]['optimizer_space']/1024**3,
                 e4.iloc[0]['optimizer_space_with_offload']/1024**3]
  colors = [
    '#117733',
    '#44AA99',
    '#999933',
    '#DDCC77',
    '#88CCEE',
  ]

  ax[1].bar(labels, weight, width, label='Weight', color=colors[0])
  ax[1].bar(labels, act_space, width, bottom=weight,
            label='Activation', color=colors[1])
  ax[1].bar(labels, weight_grad, width,
            bottom=[sum(x) for x in zip(weight, act_space)],
            label='Weight gradients', color=colors[2])
  ax[1].bar(labels, act_grad, width,
            bottom=[sum(x) for x in zip(weight, act_space, weight_grad)],
            label='Activation gradients', color=colors[3])
  ax[1].bar(labels, optim_space, width,
            bottom=[sum(x) for x in zip(weight, act_space, weight_grad, act_grad)],
            label='Optimizer space', color=colors[4])

  ax[1].set_ylabel('Size, GB', fontsize=12)
  ax[1].set_ylim([mmin, mmax])
  plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor", fontsize=11)
  ax[1].legend(loc='lower center', bbox_to_anchor=(0.45, -0.65),
               fancybox=True, shadow=True, ncol=2, alignment='center',
               fontsize=10)
  ax[1].set_title('HBM consumption', fontsize=12)

  fig.savefig(args.fig10, dpi=300, transparent=False, bbox_inches="tight")


  # Create a table of information
  with open(args.tab4, 'w') as fd:
    print('Name,time,mem,mfu,tp,pp,dp,mbs,ppint,recompute,tp_comm,redo,tpo,dpo,optshard,fused,wo,ao,oo',
          file=fd)
    for name, frame in [('Baseline', e1), ('SeqPar', e2), ('SwOpts', e3),
                        ('HwOff', e4)]:
      print(f'{name},', file=fd, end='')
      print(f'{frame.iloc[0]["total_time"]:.02f}s,', file=fd, end='')
      print(f'{frame.iloc[0]["proc_mem_tier1_cap_req"] / 1024**3:.02f}GiB,', file=fd, end='')
      print(f'{frame.iloc[0]["total_efficiency"] * 100:.02f}%,', file=fd, end='')
      print(f'{frame.iloc[0]["tensor_par"]},', file=fd, end='')
      print(f'{frame.iloc[0]["pipeline_par"]},', file=fd, end='')
      print(f'{frame.iloc[0]["data_par"]},', file=fd, end='')
      print(f'{frame.iloc[0]["microbatch_size"]},', file=fd, end='')
      print(f'{frame.iloc[0]["pipeline_interleaving"]},', file=fd, end='')
      print(f'{frame.iloc[0]["activation_recompute"]},', file=fd, end='')
      print(f'{frame.iloc[0]["tensor_par_comm_type"]},', file=fd, end='')
      print(f'{frame.iloc[0]["seq_par_ag_redo"]},', file=fd, end='')
      print(f'{frame.iloc[0]["tensor_par_overlap"]},', file=fd, end='')
      print(f'{frame.iloc[0]["data_par_overlap"]},', file=fd, end='')
      print(f'{frame.iloc[0]["optimizer_sharding"]},', file=fd, end='')
      print(f'{frame.iloc[0]["fused_activation"]},', file=fd, end='')
      print(f'{frame.iloc[0]["weight_offload"]},', file=fd, end='')
      print(f'{frame.iloc[0]["activations_offload"]},', file=fd, end='')
      print(f'{frame.iloc[0]["optimizer_offload"]},', file=fd, end='')
      print('', file=fd)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('input', help='input file')
  ap.add_argument('fig10', help='Figure 10 output file')
  ap.add_argument('tab4', help='Table 4 output file')
  sys.exit(main(ap.parse_args()))
