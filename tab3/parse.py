#!/usr/bin/env python3

import argparse
import calculon
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tol_colors as tc



def main(args):
  # Copied from __init__.py
  nvls = [8]
  mem1s = [(20, 3072), (40, 3072), (80, 3072), (120, 3072)]
  mem2s = [(0, 100), (256, 100), (512, 100), (1024, 100)]
  apps = ['175B', '530B', '1T']
  def getSystemName(nvl, mem1, mem2):
    return f'h100_m{mem1[0]}_b{mem1[1]}_d{mem2[0]}_c{mem2[1]}_nvl{nvl}_ib50'
  def getGpuPrice(mem1, mem2):
    hbm3 = {
      20: 2250,
      40: 5000,
      80: 10000,
      120: 20000,
    }
    ddr5 = {
      0: 0,
      256: 2500,
      512: 10000,
      1024: 20000,
    }
    return 20000 + hbm3[mem1[0]] + ddr5[mem2[0]]
  def getSystemSizes(mem1, mem2):
    gpu_price = getGpuPrice(mem1, mem2)
    max_system_price = 125e6
    max_gpus = int(max_system_price / gpu_price)
    system_sizes = range(8, max_gpus + 1, 8)
    return system_sizes[-(512 // 8):]

  # Creates the dataframe to hold the table
  cols = ['HBM3', 'DDR5', 'Price', 'Max GPUs']
  for app in apps:
    cols.append(f'{app}-GPUs')
    cols.append(f'{app}-Perf')
    cols.append(f'{app}-Perf/$')
  df = pd.DataFrame(columns=cols)

  # Reads in the data, creates the table text
  nvl = nvls[0]
  for mem2 in mem2s:
    for mem1 in mem1s:
      sys_name = getSystemName(nvl, mem1, mem2)
      gpu_price = getGpuPrice(mem1, mem2)
      gpu_price_str = f'${round(gpu_price/1000, 1)}k'
      if gpu_price_str.endswith('.0k'):
        gpu_price_str = gpu_price_str.replace('.0k', 'k')
      row = {
        'HBM3': mem1[0],
        'DDR5': mem2[0],
        'Price': gpu_price_str,
        'Max GPUs': getSystemSizes(mem1, mem2)[-1]
      }
      for app in apps:
        # Finds the top performer
        best_perf = 0
        best_size = 0
        for size in getSystemSizes(mem1, mem2):
          run_name = f'tab3-{app}_{size}_{sys_name}'
          run_output = os.path.join(args.directory, f'{run_name}.json.gz')
          data = calculon.read_json_file(run_output)
          if '0' in data:
            perf = data['0']['stats']['sample_rate']
            if perf > best_perf:
              best_perf = perf
              best_size = size
        # Gets data for the top performer
        run_name = f'tab3-{app}_{best_size}_{sys_name}'
        run_output = os.path.join(args.directory, f'{run_name}.json.gz')
        data = calculon.read_json_file(run_output)
        num_procs = data['0']['execution']['num_procs']
        assert num_procs == best_size
        used_system_price = num_procs * gpu_price
        perf = data['0']['stats']['sample_rate']
        assert perf == best_perf
        row[f'{app}-GPUs'] = num_procs
        row[f'{app}-Perf'] = round(perf)
        row[f'{app}-Perf/$'] = round(perf / used_system_price * 1e8)
      new = pd.DataFrame.from_records([row])
      df = pd.concat([df, new], ignore_index = True)

  # Write the output CSV file
  df.to_csv(args.output)
  print(df)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('directory', help='input and output directory')
  ap.add_argument('output', help='output file')
  sys.exit(main(ap.parse_args()))
