import os
import sys
import taskrun

H = os.path.dirname(os.path.abspath(__file__))

class Tab2():

  def __init__(self):
    self.output = os.path.join(H, 'output')

  def createTasks(self, executor):
    os.makedirs(self.output, exist_ok=True)

    # Creates the validation task
    calc_name = 'tab2'
    stats_file = os.path.join(self.output, 'stats.json')
    calc_log = os.path.join(self.output, 'validation.log')
    calc_cmd = (f'{executor.calcBin} llm-validation --base_dir '
                f'{executor.calcDir}')
    calc_task = executor.createTask('MiscProcess', calc_name, calc_cmd,
                                    calc_log)
    calc_task.add_condition(taskrun.FileModificationCondition(
      [], [calc_log]))

    # Creates the parsing task
    parser = os.path.join(H, 'parse.sh')
    assert os.path.exists(parser)
    table_file = os.path.join(self.output, 'table.txt')
    parse_name = f'{calc_name}_parse'
    parse_cmd = f'{parser} {calc_log} {table_file}'
    parse_log = os.path.join(self.output, f'{parse_name}.log')
    parse_task = executor.createTask('MiscProcess', parse_name, parse_cmd,
                                     parse_log)
    parse_task.add_condition(taskrun.FileModificationCondition(
      [calc_log], [table_file]))
    parse_task.add_dependency(calc_task)
