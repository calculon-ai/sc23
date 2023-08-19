import calculon
import os
import sys
import taskrun

H = os.path.dirname(os.path.abspath(__file__))

class Fig5():

  def __init__(self):
    self.output = os.path.join(H, 'output')

  def createTasks(self, executor):
    os.makedirs(self.output, exist_ok=True)

    for datatype, system in [
        #('float16', 'a100_80g'),
        ('float16', 'h100_80g_nvl8'),
        #('float8', 'h100_80g_nvl8')
    ]:
      # Creates the all executions task
      sys_file = os.path.join(os.environ['CALC'], 'systems', f'{system}.json')
      gpt3_175B = os.path.join(os.environ['CALC'], 'models', 'gpt3-175B.json')
      run_name = f'fig5-all-executions-{datatype}-{system}'
      run_log = os.path.join(self.output, f'{run_name}.log')
      run_output = os.path.join(self.output, f'{run_name}.csv.gz')
      run_task = executor.createAllExecutionsTask(
        run_name, gpt3_175B, 4096, 2340, datatype, sys_file, run_output,
        'both', run_log)
      run_task.add_condition(taskrun.FileModificationCondition(
        [sys_file, gpt3_175B], [run_output]))

      # Creates the plotting task
      plotter = os.path.join(H, 'fig5.py')
      assert os.path.exists(plotter)
      plot_file = os.path.join(self.output, f'{run_name}.png')
      plot_name = f'{run_name}_plot'
      plot_cmd = f'{plotter} {run_output} {plot_file}'
      plot_log = os.path.join(self.output, f'{plot_name}.log')
      plot_task = executor.createTask('MiscProcess', plot_name, plot_cmd,
                                      plot_log)
      plot_task.add_condition(taskrun.FileModificationCondition(
        [run_output], [plot_file]))
      plot_task.add_dependency(run_task)
