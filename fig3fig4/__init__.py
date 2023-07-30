import calculon
import os
import sys
import taskrun

H = os.path.dirname(os.path.abspath(__file__))

class Fig3Fig4():

  def __init__(self):
    self.output = os.path.join(H, 'output')

  def createTasks(self, executor):
    os.makedirs(self.output, exist_ok=True)

    # Creates the system file task
    a100_80g = os.path.join(os.environ['CALC'], 'systems', 'a100_80g.json')
    a100_big = os.path.join(self.output, 'a100_big.json')
    def createSystemFile():
      sys = calculon.read_json_file(a100_80g)
      sys['mem1']['GiB'] = 99999999
      sys['mem2']['GiB'] = 0
      sys['networks'][0]['size'] = 4096
      calculon.write_json_file(sys, a100_big)
    sys_task = executor.createFunctionTask('fig3fig4-sys-file',
                                           createSystemFile)
    sys_task.add_condition(taskrun.FileModificationCondition(
      [a100_80g], [a100_big]))

    # Creates the all executions task
    megatron_1T = os.path.join(os.environ['CALC'], 'models', 'megatron-1T.json')
    run_name = 'fig3fig4-all-executions'
    run_log = os.path.join(self.output, f'{run_name}.log')
    run_output = os.path.join(self.output, f'{run_name}.csv.gz')
    run_task = executor.createAllExecutionsTask(
      run_name, megatron_1T, 4096, 3072, 'float16', a100_big, run_output,
      'both', run_log)
    run_task.add_condition(taskrun.FileModificationCondition(
      [a100_big, megatron_1T], [run_output]))

    # Creates the plotting tasks
    for fig in [3, 4]:
      plotter = os.path.join(H, f'fig{fig}.py')
      assert os.path.exists(plotter)
      fig_file = os.path.join(self.output, f'fig{fig}.png')
      fig_name = f'{run_name}_fig{fig}'
      fig_cmd = f'{plotter} {run_output} {fig_file}'
      fig_log = os.path.join(self.output, f'{fig_name}.log')
      fig_task = executor.createTask('MiscProcess', fig_name, fig_cmd, fig_log)
      fig_task.add_condition(taskrun.FileModificationCondition(
        [run_output], [fig_file]))
      fig_task.add_dependency(run_task)
