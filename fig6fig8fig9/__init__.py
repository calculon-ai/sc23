import calculon
import copy
import os
import sys
import taskrun

H = os.path.dirname(os.path.abspath(__file__))

class Fig6Fig8Fig9():

  def __init__(self):
    self.output = os.path.join(H, 'output')

  def createTasks(self, executor):
    os.makedirs(self.output, exist_ok=True)

    # Defines the multi-dimensional sweep parameters
    nvls = [8]
    sizes = sorted(list(set(list(range(8, 8*1024+1, 8)))))
    mem2s = [512]
    g2cs = [0, 100]
    apps = [
      ['175B', 'gpt3-175B', '2340'],
      ['530B', 'turing-530B', '2520'],
      ['1T', 'megatron-1T', '3072']]

    # Gathers input files
    sys_tmpl = os.path.join(os.environ['CALC'], 'systems', 'h100_80g_nvl8.json')
    assert os.path.exists(sys_tmpl)
    for index, app in enumerate(apps):
      _, fullname, _ = app
      app_file = os.path.join(os.environ['CALC'], 'models', f'{fullname}.json')
      assert os.path.exists(app_file)
      apps[index][1] = app_file

    # Creates the system files task
    def getSystemName(nvl, mem2, g2c):
      return f'h100m80_ddr{mem2}_g2c{g2c}_nvl{nvl}_ib50'
    def getSystemFile(nvl, mem2, g2c):
      sys_name = getSystemName(nvl, mem2, g2c)
      return os.path.join(self.output, f'{sys_name}.json')
    def getSystemFiles():
      return [getSystemFile(nvl, mem2, g2c)
              for nvl in nvls
              for mem2 in mem2s
              for g2c in g2cs]
    def createSystemFiles():
      tmpl = calculon.read_json_file(sys_tmpl)
      for nvl in nvls:
        for mem2 in mem2s:
          for g2c in g2cs:
            sys = copy.deepcopy(tmpl)
            sys['networks'][0]['size'] = nvl
            if g2c > 0:
              sys['mem2']['GiB'] = mem2
              sys['mem2']['GBps'] = g2c
            else:
              sys['mem2']['GiB'] = 0
              sys['mem2']['GBps'] = 1
            sys_file = getSystemFile(nvl, mem2, g2c)
            calculon.write_json_file(sys, sys_file)
    sys_task = executor.createFunctionTask('fig6fig8fig9-sys-files',
                                           createSystemFiles)
    sys_task.add_condition(taskrun.FileModificationCondition(
      [sys_tmpl], getSystemFiles()))

    # Creates all the optimal execution tasks
    run_tasks = []
    run_outputs = []
    for nvl in nvls:
      for mem2 in mem2s:
        for g2c in g2cs:
          sys_name = getSystemName(nvl, mem2, g2c)
          sys_file = getSystemFile(nvl, mem2, g2c)
          for app, app_file, gbs in apps:
            for size in sizes:
              run_name = f'fig6fig8fig9-{app}_{size}_{sys_name}'
              run_log = os.path.join(self.output, f'{run_name}.log')
              run_output = os.path.join(self.output, f'{run_name}.json.gz')
              run_task = executor.createOptimalExecutionTask(
                run_name, app_file, size, gbs, 'float16', sys_file, run_output,
                10, 'both', run_log)
              run_task.add_condition(taskrun.FileModificationCondition(
                [sys_file, app_file], [run_output]))
              run_task.add_dependency(sys_task)
              run_tasks.append(run_task)
              run_outputs.append(run_output)

    # Creates the plotting task for the figure
    plotter = os.path.join(H, 'figs.py')
    assert os.path.exists(plotter)
    fig_files = []
    fig_files = [os.path.join(self.output, f'fig{fig}.pdf')
                 for fig in [6, 8, 9]]
    fig_name = f'{run_name}_figs'
    fig_cmd = f'{plotter} {self.output}'
    fig_log = os.path.join(self.output, f'{fig_name}.log')
    fig_task = executor.createTask('MiscProcess', fig_name, fig_cmd,
                                   fig_log)
    fig_task.add_condition(taskrun.FileModificationCondition(
      run_outputs, fig_files))
    for run_task in run_tasks:
      fig_task.add_dependency(run_task)
