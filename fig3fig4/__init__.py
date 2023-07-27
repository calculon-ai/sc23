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

    # Creates the plotting task for fig3
    plotter = os.path.join(H, 'fig3.py')
    assert os.path.exists(plotter)
    fig3_file = os.path.join(self.output, 'fig3.png')
    fig3_name = f'{run_name}_fig3'
    fig3_cmd = f'{plotter} {run_output} {fig3_file}'
    fig3_log = os.path.join(self.output, f'{fig3_name}.log')
    fig3_task = executor.createTask('MiscProcess', fig3_name, fig3_cmd,
                                    fig3_log)
    fig3_task.add_condition(taskrun.FileModificationCondition(
      [run_output], [fig3_file]))
    fig3_task.add_dependency(run_task)

    # Creates the plotting task for fig4
    plotter = os.path.join(H, 'fig4.py')
    assert os.path.exists(plotter)
    fig4_file = os.path.join(self.output, 'fig4.png')
    fig4_name = f'{run_name}_fig4'
    fig4_cmd = f'{plotter} {run_output} {fig4_file}'
    fig4_log = os.path.join(self.output, f'{fig4_name}.log')
    fig4_task = executor.createTask('MiscProcess', fig4_name, fig4_cmd,
                                    fig4_log)
    fig4_task.add_condition(taskrun.FileModificationCondition(
      [run_output], [fig4_file]))
    fig4_task.add_dependency(run_task)
