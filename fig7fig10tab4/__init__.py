import calculon
import copy
import os
import sys
import taskrun

H = os.path.dirname(os.path.abspath(__file__))

class Fig7Fig10Tab4():

  def __init__(self):
    self.output = os.path.join(H, 'output')

  def createTasks(self, executor):
    os.makedirs(self.output, exist_ok=True)

    # Creates the system file task
    h100_80g_nvl8 = \
      os.path.join(os.environ['CALC'], 'systems', 'h100_80g_nvl8.json')
    h100_80g_nvl4k_infinite_off = \
      os.path.join(self.output, 'h100_80g_nvl4k_infinite_off.json')
    h100_80g_nvl4k_real_off = \
      os.path.join(self.output, 'h100_80g_nvl4k_real_off.json')
    def createSystemFiles():
      base = calculon.read_json_file(h100_80g_nvl8)
      # Mythical offloading system
      inf = copy.deepcopy(base)
      inf['mem2']['GiB'] = 99999999
      inf['mem2']['GBps'] = 99999999
      inf['networks'][0]['size'] = 4096
      calculon.write_json_file(inf, h100_80g_nvl4k_infinite_off)
      # Real offloading system
      real = copy.deepcopy(base)
      real['mem2']['GiB'] = 512
      real['mem2']['GBps'] = 100
      real['networks'][0]['size'] = 4096
      calculon.write_json_file(real, h100_80g_nvl4k_real_off)

    sys_task = executor.createFunctionTask('fig7fig10tab4-sys-file',
                                           createSystemFiles)
    sys_task.add_condition(taskrun.FileModificationCondition(
      [h100_80g_nvl8], [h100_80g_nvl4k_infinite_off, h100_80g_nvl4k_real_off]))

    # Creates the all executions task
    megatron_1T = os.path.join(os.environ['CALC'], 'models', 'megatron-1T.json')
    run_tasks = {}
    run_outputs = {}
    for off, sys_file in [('inf', h100_80g_nvl4k_infinite_off),
                          ('real', h100_80g_nvl4k_real_off)]:
      run_name = f'fig7fig10tab4-all-executions-{off}'
      run_log = os.path.join(self.output, f'{run_name}.log')
      run_output = os.path.join(self.output, f'{run_name}.csv.gz')
      run_task = executor.createAllExecutionsTask(
        run_name, megatron_1T, 4096, 3072, 'float16', sys_file,
        run_output, 'both', run_log)
      run_task.add_condition(taskrun.FileModificationCondition(
        [sys_file, megatron_1T], [run_output]))
      run_task.add_dependency(sys_task)
      run_tasks[off] = run_task
      run_outputs[off] = run_output

    # Creates the plotting task for fig7
    plotter = os.path.join(H, 'fig7.py')
    assert os.path.exists(plotter)
    fig7_file = os.path.join(self.output, 'fig7.png')
    fig7_name = 'fig7fig10tab4_fig7'
    all_run_outputs = ' '.join(list(run_outputs.values()))
    fig7_cmd = f'{plotter} {all_run_outputs} {fig7_file}'
    fig7_log = os.path.join(self.output, f'{fig7_name}.log')
    fig7_task = executor.createTask('MiscProcess', fig7_name, fig7_cmd,
                                    fig7_log)
    fig7_task.add_condition(taskrun.FileModificationCondition(
      list(run_outputs.values()), [fig7_file]))
    for run_task in run_tasks.values():
      fig7_task.add_dependency(run_task)

    # Creates the plotting task for fig10 and table creator for tab4
    plotter = os.path.join(H, 'fig10tab4.py')
    assert os.path.exists(plotter)
    fig10_file = os.path.join(self.output, 'fig10.png')
    tab4_file = os.path.join(self.output, 'tab4.csv')
    fig10tab4_name = 'fig7fig10tab4_fig10tab4'
    fig10tab4_cmd = f'{plotter} {run_outputs["real"]} {fig10_file} {tab4_file}'
    fig10tab4_log = os.path.join(self.output, f'{fig10tab4_name}.log')
    fig10tab4_task = executor.createTask('MiscProcess', fig10tab4_name,
                                         fig10tab4_cmd, fig10tab4_log)
    fig10tab4_task.add_condition(taskrun.FileModificationCondition(
      [run_outputs['real']], [fig10_file, tab4_file]))
    fig10tab4_task.add_dependency(run_tasks['real'])
