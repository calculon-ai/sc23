import os
import psutil
import taskrun
import tempfile


class Executor():
  """ This class allows for a flexible framework of executing parallel on a
  distributed cluster of machines. It is assumed all machines connect to the
  same file system.
  """

  SupportedModes = ['local_full', 'local_8', 'nvlsf']

  def __init__(self, calc_dir, execution_mode, failure_mode = 'passive_fail'):
    self._calc_dir = calc_dir
    self._mode = execution_mode
    if self._mode.startswith('local'):
      if self._mode == 'local_full':
        self._parallelExecutionCores = psutil.cpu_count(logical=False)
        self._singleExecutionSlots = 1
        self._parallelExecutionSlots = psutil.cpu_count(logical=False)
        self._miscTaskSlots = 1
      elif self.mode == 'local_8':
        self._parallelExecutionCores = 8
        self._singleExecutionSlots = 1
        self._parallelExecutionSlots = 8
        self._miscTaskSlots = 1
      self._tm = taskrun.standard_task_manager(
        track_cpus = True,
        max_cpus = self._parallelExecutionSlots,
        track_memory = True,
        max_memory = -1,
        cleanup_files = True,
        failure_mode = failure_mode)
    elif self._mode == 'nvlsf':
      self._maxLsfSlotsPerUser = 600
      self._lsfParallelExecutionCores = 8
      self._tm = taskrun.standard_task_manager(
        track_cpus = True,
        max_cpus = self._maxLsfSlotsPerUser,
        track_memory = False,
        cleanup_files = True,
        failure_mode = failure_mode)
      self._parallelExecutionCores = self._lsfParallelExecutionCores
      self._singleExecutionSlots = 1
      self._parallelExecutionSlots = self._lsfParallelExecutionCores
      self._miscTaskSlots = 1
    else:
      assert False

  @property
  def calcDir(self):
    return self._calc_dir

  @property
  def calcBin(self):
    return os.path.join(self._calc_dir, 'bin', 'calculon')

  @property
  def parallelExecutionCores(self):
    return self._parallelExecutionCores

  def createTask(self, task_type, name, command, log):
    """Generates an appropriately created taskrun.Task.

    Args:
      task_type (str): 'SingleExecution', 'OptimalExecution', 'AllExecutions',
                       or 'MiscProcess'
      name (str): Name of the task
      command (str): Command string of the task that will be executed on the
        command line
      log (str): The log file that is used for stdout. Stderr should be:
        f'{log}.err'

    Returns:
      task (Task): The created task.

    *** task_type == 'SingleExecution':
    This runs a single calculon execution. 2GB of memory and 1ms.

    *** task_type == 'OptimalExecution':
    This runs an optimal execution search using 'command'. This is a subprocess
    that executes across multiple logical cores using the multiprocessing
    library. It generally requires up to 8GB of RAM and takes up to 45 minutes
    but often much less. The command already tells Calculon to only use
    'parallelExecutionCores' but if required, the reformated command should
    inform the cluster scheduler of the same.

    *** task_type == 'AllExecutions':
    This runs lists all executions using 'command'. This is a subprocess
    that executes across multiple logical cores using the multiprocessing
    library. It generally requires up to 32GB of RAM and takes up to 45 minutes
    but often much less. The command already tells Calculon to only use
    'parallelExecutionCores' but if required, the reformated command should
    inform the cluster scheduler of the same.

    *** task_type == 'MiscProcess':
    This is single threaded misc task that uses 4GB and should finish in only a
    few seconds or minutes.
    """
    if task_type == 'SingleExecution':
      gb = 4
      hr = 1
      slots = self._singleExecutionSlots
    elif task_type == 'OptimalExecution':
      gb = 8
      hr = 8
      slots = self._parallelExecutionSlots
    elif task_type == 'AllExecutions':
      gb = 32
      hr = 8
      slots = self._parallelExecutionSlots
    elif task_type == 'MiscProcess':
      gb = 16
      hr = 1
      slots = self._miscTaskSlots
    else:
      assert False, 'bad programmer :('
    return self.createProcessTask(name, command, log, gb, hr, slots)

  def createProcessTask(self, name, command, log, gb, hr, slots):
    if self._mode.startswith('local'):
      cmd = command
      log = log
    elif self._mode == 'nvlsf':
      cmd = (
        '/home/nv/bin/qsub '
        '-P research_networking_misc '
        '-m rel75 '
        '-env all '
        f'-q o_cpu_{gb}G_{hr}H '
        '-K '
        f'-J {name} ')
      if slots > 1:
        cmd += f'-n {slots} -R \'span[hosts=1]\' '
      cmd += (
        f'-oo {log} '
        f'-eo {log}.err '
        f'{command} ')
      log = ''
    else:
      assert False, 'bad programmer :('

    task = taskrun.ProcessTask(self._tm, name, cmd)
    if log != '':
      task.stdout_file = log
      task.stderr_file = log + '.err'

    if self._mode.startswith('local'):
      task.resources = {
        'cpus': slots,
        'mem': gb
      }
    elif self._mode == 'nvlsf':
      task.resources = {
        'cpus': slots
      }
    else:
      assert False, 'bad programmer :('
    return task

  def createFunctionTask(self, name, func, *args, **kwargs):
    task = taskrun.FunctionTask(self._tm, name, func, *args, **kwargs)
    if self._mode.startswith('local'):
      task.resources = {
        'cpus': 1,
        'mem': 0
      }
    elif self._mode == 'nvlsf':
      task.resources = {
        'cpus': 1
      }
    else:
      assert False, 'bad programmer :('
    return task

  def run_tasks(self):
    return self._tm.run_tasks()

  def test(self, test_command):
    def tfunc(a, b, c):
      _, log = tempfile.mkstemp(suffix='.log')
      with open(log, 'w') as fd:
        print(a + b + c, file=fd)
    self.createFunctionTask('tfunc', tfunc, 3, 4, 5)
    for task_type in ['SingleExecution', 'OptimalExecution', 'MiscProcess']:
      _, log = tempfile.mkstemp(suffix='.log')
      os.remove(log)
      name = f'{task_type}Test'
      cmd = f'{test_command} -h'
      task = self.createTask(task_type, name, cmd, log)
    if not self.run_tasks():
      print('Task execution failed')
      return -1
    else:
      return 0

  def createSingleExecutionTask(self, name, app, exe, sys, stats, log):
    cmd = (
      f'{self.calcBin} '
      'llm '
      f'{app} '
      f'{exe} '
      f'{sys} '
      f'{stats} '
    )
    return self.createTask('SingleExecution', name, cmd, log)

  def createOptimalExecutionTask(self, name, app, num_procs, max_batch_size,
                                 datatype, sys, output, top_n, fused_act, log):
    cmd = (
      f'{self.calcBin} '
      'llm-optimal-execution '
      f'{app} '
      f'{num_procs} '
      f'{max_batch_size} '
      f'{datatype} '
      f'{sys} '
      f'{output} '
      f'-c {self._parallelExecutionCores} '
      f'-n '
      f'-m '
      f'-f {fused_act} '
      f'-t {top_n} '
    )
    return self.createTask('OptimalExecution', name, cmd, log)

  def createAllExecutionsTask(self, name, app, num_procs, max_batch_size,
                              datatype, sys, output, fused_act, log):
    cmd = (
      f'{self.calcBin} '
      'llm-all-executions '
      f'{app} '
      f'{num_procs} '
      f'{max_batch_size} '
      f'{datatype} '
      f'{sys} '
      f'{output} '
      f'-c {self._parallelExecutionCores} '
      f'-n '
      f'-f {fused_act} '
    )
    return self.createTask('AllExecutions', name, cmd, log)
