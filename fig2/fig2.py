#!/usr/bin/env python3

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args):
  with open(args.input) as fd:
    stats = json.load(fd)

  fig, ax = plt.subplots(1, 2, figsize=(6, 2))
  width = 0.7
  tmin=0
  tmax=20
  mmin=0
  mmax=20

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
    ""
  ]

  fw_time = [stats['fw_time']]
  bw_time = [stats['bw_time']]
  optim_time = [stats['optim_step_time']]
  recomp_time = [stats['recompute_time']]
  bubble_time = [stats['bubble_time']]
  tp_time = [stats['tp_comm_exposed_time'] + stats['recomm_exposed_time']]
  pp_time = [stats['pp_comm_exposed_time']]
  dp_time = [stats['dp_comm_exposed_time']]

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

  ax[0].set_ylabel('Time, s')
  ax[0].set_ylim([tmin, tmax])
  ax[0].set_xlim(-1, 1)
  ax[0].legend(loc='lower center', bbox_to_anchor=(0.45, -0.8),
               fancybox=True, shadow=True, ncol=2)
  ax[0].set_title('Batch time')

  weight = [stats['weight_space']/1024**3]
  act_space = [stats['act_space']/1024**3 + stats['act_checkpoint_size']/1024**3]
  weight_grad = [stats['weight_grad_space']/1024**3]
  act_grad = [stats['act_grad_space']/1024**3]
  optim_space = [stats['optimizer_space']/1024**3]
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

  ax[1].set_ylabel('Size, GB')
  ax[1].set_ylim([mmin, mmax])
  ax[1].set_xlim(-1, 1)
  ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.8),
               fancybox=True, shadow=True, ncol=1)
  ax[1].set_title('HBM consumption')
  plt.subplots_adjust(wspace=0.4)
  fig.savefig(args.output, dpi=300, transparent=False, bbox_inches="tight")


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('input', help='input file')
  ap.add_argument('output', help='output file')
  main(ap.parse_args())
