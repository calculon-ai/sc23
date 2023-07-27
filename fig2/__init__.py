import os
import sys
import taskrun

H = os.path.dirname(os.path.abspath(__file__))

class Fig2():

  def __init__(self):
    self.output = os.path.join(H, 'output')

  def createTasks(self, executor):
    assert os.path.isdir(os.path.join(H, 'input')), \
      f'{os.path.join(H, "input")}'
    os.makedirs(self.output, exist_ok=True)

    # Gathers input files
    app_file = os.path.join(os.environ['CALC'], 'models', 'gpt3-175B.json')
    exe_file = os.path.join(H, 'input', 'exe.json')
    sys_file = os.path.join(os.environ['CALC'], 'systems', 'a100_80g.json')
    assert os.path.exists(app_file)
    assert os.path.exists(exe_file)
    assert os.path.exists(sys_file)

    # Runs a single execution of GPT3-175B
    calc_name = 'fig2_gpt3-175B'
    stats_file = os.path.join(self.output, 'stats.json')
    calc_log = os.path.join(self.output, f'{calc_name}.log')
    calc_task = executor.createSingleExecutionTask(
      calc_name, app_file, exe_file, sys_file, stats_file, calc_log)
    calc_task.add_condition(taskrun.FileModificationCondition(
      [app_file, exe_file, sys_file], [stats_file]))

    # Creates the plotting task
    plotter = os.path.join(H, 'fig2.py')
    assert os.path.exists(plotter)
    plot_file = os.path.join(self.output, 'plot.png')
    plot_name = f'{calc_name}_plot'
    plot_cmd = f'{plotter} {stats_file} {plot_file}'
    plot_log = os.path.join(self.output, f'{plot_name}.log')
    plot_task = executor.createTask('MiscProcess', plot_name, plot_cmd,
                                    plot_log)
    plot_task.add_condition(taskrun.FileModificationCondition(
      [stats_file], [plot_file]))
    plot_task.add_dependency(calc_task)
