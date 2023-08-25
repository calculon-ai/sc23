#!/usr/bin/env python3

import argparse
import calculon
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def main(args):
  # Defines the multi-dimensional sweep parameters
  nvls = [8]
  sizes = sorted(list(set(list(range(8, 8*1024+1, 8)))))
  mem2s = [512]
  g2cs = [0, 100]
  apps = ['175B', '530B', '1T']

  def getSystemName(nvl, mem2, g2c):
    return f'h100m80_ddr{mem2}_g2c{g2c}_nvl{nvl}_ib50'

  colors = {
    '175B': '#BB5566',
    '530B': '#DDAA33',
    '1T': '#004488'
  }
  labels = {
    '175B': 'GPT3 175B',
    '530B': 'Turing-NLG 530B',
    '1T': 'Megatron-1T'
  }

  # Makes all plots
  for nvl in nvls:
    for mem2 in mem2s:
      for g2c in g2cs:
        sys_name = getSystemName(nvl, mem2, g2c)

        # Creates plot structure
        fig, ax = plt.subplots(1, 3, figsize=(7.5, 3.5))
        if g2c == 0:
          title = 'LLM training scalability (no offloading)'
        else:
          title = f'LLM training scalability ({g2c} GB/s offloading)'
        fig.suptitle(title, fontsize=18, y=1.04)

        for ai, app in enumerate(apps):
          perfs = []
          max_point = 0
          for size in sizes:
            # Reads run data
            run_name = f'fig6fig8fig9-{app}_{size}_{sys_name}'
            run_output = os.path.join(args.directory, f'{run_name}.json.gz')
            data = calculon.read_json_file(run_output)

            # Parses run data
            if '0' in data:
              perf = data['0']['stats']['sample_rate']
              eff = data['0']['stats']['system_efficiency']
              max_point = max(perf / eff, max_point)
            else:
              perf = 0
            perfs.append(perf)

          # Formats data for plotting
          bests = np.maximum.accumulate(perfs)
          perfs = np.asarray(perfs)
          sizes = np.asarray(sizes)

          # Perfect scaling
          ax[ai].plot(sizes, sizes / np.max(sizes), '--', linewidth=1,
                      color='k')

          # Scatter plot of relative performance
          ax[ai].scatter(sizes, perfs / max_point, sizes=[25]*len(sizes),
                         marker='1', linewidths=0.7, color=colors[app])

          # Best performance accumulation
          ax[ai].plot(sizes, bests / max_point, linewidth=1, color='k')

          # Misc formatting
          ax[ai].xaxis.set_ticks(np.arange(0, max(sizes)+1, 1024))
          plt.setp(ax[ai].get_xticklabels(), rotation=45, ha='right',
                   rotation_mode='anchor', fontsize=11)
          plt.setp(ax[ai].get_yticklabels(), fontsize=11)
          ax[ai].set_title(labels[app])
          ax[ai].grid(True)

        # Labels and legend
        ax[0].set_ylabel('Relative scaling', fontsize=12)
        ax[1].set_xlabel('System size', fontsize=12)
        ax[1].plot([], [], '--', color='k', label='Perfect scaling')
        ax[1].plot([], [], '-', color='k', label='Best performance')
        ax[1].legend(loc='lower center', bbox_to_anchor=(0.4, -0.46),
                     fancybox=True, shadow=True, ncol=3, fontsize=12)

        plt.subplots_adjust(wspace=0.25)
        fi = 6 if g2c == 0 else 8
        plotfile = os.path.join(args.directory, f'fig{fi}.pdf')
        print(f'Saving {plotfile}')
        fig.savefig(plotfile, dpi=300, transparent=False, bbox_inches='tight')

  offloads = copy.copy(g2cs)
  offloads.remove(0)

  # Makes all plots
  for nvl in nvls:
    for mem2 in mem2s:
      for g2c in offloads:
        sys_name_non = getSystemName(nvl, mem2, 0)
        sys_name_off = getSystemName(nvl, mem2, g2c)

        # Creates plot structure
        fig, ax = plt.subplots(
          6, 1, figsize=(7.5, 6), sharex=True,
          gridspec_kw={'height_ratios': [1, 4, 1, 4, 1, 4]})
        title = (f'Relative performance improvement with offloading\n'
                 f'({mem2} GiB @ {g2c} GB/s)')
        fig.suptitle(title, fontsize=18, y=1)

        for ai, app in enumerate(apps):
          perfs_non = []
          perfs_off = []
          max_point_non = 0
          max_point_off = 0
          for size in sizes:
            # Reads run data
            run_name = f'fig6fig8fig9-{app}_{size}_{sys_name_non}'
            run_output = os.path.join(args.directory, f'{run_name}.json.gz')
            data_non = calculon.read_json_file(run_output)
            run_name = f'fig6fig8fig9-{app}_{size}_{sys_name_off}'
            run_output = os.path.join(args.directory, f'{run_name}.json.gz')
            data_off = calculon.read_json_file(run_output)

            # Parses run data
            if '0' in data_non:
              perf_non = data_non['0']['stats']['sample_rate']
              eff_non = data_non['0']['stats']['system_efficiency']
              max_point_non = max(perf_non / eff_non, max_point_non)
            else:
              perf_non = 0
            perfs_non.append(perf_non)
            if '0' in data_off:
              perf_off = data_off['0']['stats']['sample_rate']
              eff_off = data_off['0']['stats']['system_efficiency']
              max_point_off = max(perf_off / eff_off, max_point_off)
            else:
              perf_off = 0
            perfs_off.append(perf_off)

          # Formats data for plotting
          bests_non = np.maximum.accumulate(perfs_non) / max_point_non
          bests_off = np.maximum.accumulate(perfs_off) / max_point_off
          rels = []
          for off, non in zip(bests_off, bests_non):
            if off == 0 and non == 0:
              rel = 0
            elif off > 0 and non == 0:
              rel = 300
            elif off == 0 and non > 0:
              assert False
            elif off > 0 and non > 0:
              rel = ((off / non) - 1) * 100
            else:
              assert False
            rels.append(rel)
          rels = np.asarray(rels)
          sizes = np.asarray(sizes)

          # Best performance accumulation
          ax[2*ai].plot(sizes, rels, linewidth=1.5,
                        color=colors[app], label=labels[app])
          ax[2*ai+1].plot(sizes, rels, linewidth=1.5,
                          color=colors[app])
          ax[2*ai].set_title(labels[app], fontsize=14, y=-0.2,
                             bbox={'facecolor': 'white', 'linewidth': 0})
          ax[2*ai+1].xaxis.set_ticks(np.arange(0, max(sizes)+1, 256))

          # Combining the two axis'
          ax[2*ai].spines['bottom'].set_visible(False)
          ax[2*ai+1].spines['top'].set_visible(False)
          d = 0.25
          o = dict(marker=[(-1, -d), (1, d)], markersize=12,
                   linestyle="none", color='k', mec='k', mew=1,
                   clip_on=False)
          ax[2*ai].plot([0, 1], [0, 0], transform=ax[2*ai].transAxes, **o)
          ax[2*ai+1].plot([0, 1], [1, 1], transform=ax[2*ai+1].transAxes, **o)

          # Misc formatting
          ax[2*ai].grid(True)
          ax[2*ai].xaxis.set_ticks_position('none')
          ax[2*ai].set_yticks([300])
          ax[2*ai].set_yticklabels(['$\infty$'])
          ax[2*ai].get_yticklabels()[-1].set_fontsize(18)
          ax[2*ai+1].grid(True)
          ax[2*ai+1].set_yticks([0, 10, 20, 40, 60])
          ax[2*ai+1].set_yticklabels(['0', '10', '20', '40', '60'], fontsize=11)
          ax[2*ai].set_ylim(280, 320)
          ax[2*ai].set_xlim(0, max(sizes))
          ax[2*ai+1].set_ylim(0, 70)
          ax[2*ai+1].set_xlim(0, max(sizes))

        # Labeling
        plt.setp(ax[5].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor",  fontsize=11)
        ax[5].set_xlabel('System size', fontsize=12)
        ax[3].set_ylabel('Relative speedup, %', fontsize=12)
        plotfile = os.path.join(args.directory, f'fig9.pdf')
        print(f'Saving {plotfile}')
        fig.savefig(plotfile, dpi=300, transparent=False, bbox_inches="tight")



if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('directory', help='input and output directory')
  sys.exit(main(ap.parse_args()))
