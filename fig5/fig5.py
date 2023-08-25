#!/usr/bin/env python3

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys


def main(args):
  dfa = pandas.read_csv(args.input)
  print(f'Raw data has {dfa.shape[0]} rows')

  df = dfa #[dfa['tensor_par_overlap'] != 'ring']
  print(f'Filter data has {df.shape[0]} rows')
  num = df.shape[0]

  srs = df['sample_rate']
  srs = srs.sort_values(inplace=False, ascending=True).values
  del df

  print(f'first={srs[0]} last={srs[-1]}')

  fig, ax = plt.subplots(1, 2, figsize=(7.5, 4))
  fig.suptitle(f'{num:,} execution strategies\n'
               'for GPT3 175B on 4096 GPUs', fontsize=18)

  ax[0].hist(srs, bins=10, edgecolor='black')
  ax[0].set_xlabel('Sample rate', fontsize=12)
  ax[0].set_xticks([0, 200, 400, 600, 800, 1000])
  ax[0].set_xticklabels(['0', '200', '400', '600', '800', '1000'], fontsize=11)
  ax[0].set_ylabel('Occurances', fontsize=12)
  ax[0].set_yticks([0, 300000, 600000])
  ax[0].set_yticklabels(['0', '300k', '600k'], fontsize=11)
  ax[0].set_title('(a) Sample rate distribution', fontsize=12)

  top_n = 100
  top_srs = srs[-top_n:]
  n = len(top_srs)
  cdf = np.arange(1, n+1) / n

  ax[1].plot(top_srs, cdf, marker='.', linestyle='none')
  ax[1].set_xlabel('Sample rate', fontsize=12)
  ax[1].set_xticks([930, 970, 1010, 1050, 1090])
  ax[1].set_xticklabels(['930', '970', '1010', '1050', '1090'], fontsize=11)
  ax[1].set_ylabel('CDF', fontsize=12)
  ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
  ax[1].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
  ax[1].set_title(f'(b) Top-{top_n} sample rate CDF', fontsize=12)

  fig.tight_layout()
  fig.savefig(args.output, dpi=300, transparent=False)
  plt.close(fig)

  print('Top 100')
  for idx in range(1, 100+1):
    ridx = -1 * idx
    print(f'  {idx}: {srs[ridx]}')

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('input', help='input file')
  ap.add_argument('output', help='output file')
  sys.exit(main(ap.parse_args()))
