#!/usr/bin/env python3

import argparse
import os
import shutil
import sys

H = os.path.dirname(os.path.abspath(__file__))

# location check, environment setup, and Calculon
calc_dir = os.path.abspath(os.path.join('.', 'calc_proj'))
assert os.path.exists(calc_dir), f'Where is {calc_dir}?'
calc_bin = os.path.join(calc_dir, 'bin', 'calculon')
assert os.path.exists(calc_bin), f'Where is {calc_bin}?'
os.environ['PYTHONPATH'] = calc_dir
sys.path.append(calc_dir)
os.environ['CALC'] = calc_dir
import calculon

# This program
from executor import Executor
modules = {}
import fig2
modules['fig2'] = fig2.Fig2()
import tab2
modules['tab2'] = tab2.Tab2()
import fig3fig4
modules['fig3fig4'] = fig3fig4.Fig3Fig4()
import fig5
modules['fig5'] = fig5.Fig5()


def main(args):
  def vprint(msg):
    if args.verbose:
      print(msg)

  # Lists available items
  if args.execution_mode == 'list':
    print(f'Available items: {list(modules.keys())}')
    return

  # Creates an executor
  executor = Executor(calc_dir, args.execution_mode)
  if args.test_tasking:
    return executor.test(f'{executor.calcBin} -h')

  # Sets the executor for all modules
  for module in modules.values():
    module.executor = executor

  # Bail out if user didn't select any items
  if len(args.items) == 0:
    print(f'No items selected for execution mode \'{args.execution_mode}\'')
    return

  # Gather the tasks for all items, optional clean before
  for item in args.items:
    if args.clean:
      print(f'Clean outputs for {item}')
      shutil.rmtree(modules[item].output)
    print(f'Getting tasks for {item}')
    modules[item].createTasks(executor)

  # Run the tasks
  if not args.skip_run:
    print('Running tasks')
    executor.run_tasks()


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('execution_mode', type=str,
                  choices=['list'] + Executor.SupportedModes,
                  help='Mode of execution (\'list\' for showing items")')
  ap.add_argument('items', metavar='item', nargs='*',
                  help='An item to run')
  ap.add_argument('--clean', action='store_true',
                  help='Clean outputs before creating tasks')
  ap.add_argument('--test_tasking', action='store_true',
                  help='Test execution infrastructure by running sample tasks')
  ap.add_argument('--skip_run', action='store_true',
                  help='Don\'t run tasks, only create them')
  ap.add_argument('-v', '--verbose', action='store_true',
                  help='Verbose output')
  args = ap.parse_args()
  sys.exit(main(args))
