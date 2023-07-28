import calculon
import copy
import os
import sys
import taskrun

H = os.path.dirname(os.path.abspath(__file__))

class Tab3():

  def __init__(self):
    self.output = os.path.join(H, 'output')

  def createTasks(self, executor):
    os.makedirs(self.output, exist_ok=True)

    # Defines the multi-dimensional sweep parameters
    nvls = [8]
    mem1s = [(20, 3072), (40, 3072), (80, 3072), (120, 3072)]
    mem2s = [(0, 100), (256, 100), (512, 100), (1024, 100)]
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
    def getSystemName(nvl, mem1, mem2):
      return f'h100_m{mem1[0]}_b{mem1[1]}_d{mem2[0]}_c{mem2[1]}_nvl{nvl}_ib50'
    def getSystemFile(nvl, mem1, mem2):
      sys_name = getSystemName(nvl, mem1, mem2)
      return os.path.join(self.output, f'{sys_name}.json')
    def getSystemFiles():
      return [getSystemFile(nvl, mem1, mem2)
              for nvl in nvls
              for mem1 in mem1s
              for mem2 in mem2s]
    def createSystemFiles():
      tmpl = calculon.read_json_file(sys_tmpl)
      for nvl in nvls:
        for mem1 in mem1s:
          for mem2 in mem2s:
            sys = copy.deepcopy(tmpl)
            sys['networks'][0]['size'] = nvl
            sys['mem1']['GiB'] = mem1[0]
            sys['mem1']['GBps'] = mem1[1]
            sys['mem2']['GiB'] = mem2[0]
            sys['mem2']['GBps'] = mem2[1]
            sys_file = getSystemFile(nvl, mem1, mem2)
            calculon.write_json_file(sys, sys_file)
    sys_task = executor.createFunctionTask('tab3-sys-files',
                                           createSystemFiles)
    sys_task.add_condition(taskrun.FileModificationCondition(
      [sys_tmpl], getSystemFiles()))

    # Using a pricing model, this determines the system sizes for each system
    def getSystemSizes(mem1, mem2):
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
      gpu_price = 20000 + hbm3[mem1[0]] + ddr5[mem2[0]]
      max_system_price = 125e6
      max_gpus = int(max_system_price / gpu_price)
      system_sizes = range(8, max_gpus + 1, 8)
      return system_sizes[-(512 // 8):]

    # Creates all the optimal execution tasks
    run_tasks = []
    run_outputs = []
    for nvl in nvls:
      for mem1 in mem1s:
        for mem2 in mem2s:
          sys_name = getSystemName(nvl, mem1, mem2)
          sys_file = getSystemFile(nvl, mem1, mem2)
          for app, app_file, gbs in apps:
            for size in getSystemSizes(mem1, mem2):
              run_name = f'tab3-{app}_{size}_{sys_name}'
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

    # Need to create table from data
