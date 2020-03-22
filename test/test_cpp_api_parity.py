import os

import torch.testing._internal.common_utils as common
import torch.testing._internal.common_nn as common_nn
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity import module_impl_check, functional_impl_check, sample_module, sample_functional

print_cpp_source = True

devices = ['cpu', 'cuda']

# yf225 TODO: need to add proper checks and expectations when people:
# 1. Add a new test to a module already supported by C++ API (i.e. parity table has entry for it, and the parity bit is yes)
#   a) add a flag `test_cpp_api_parity` to the dict to be able to turn off test as needed
# 2. Add a new test for a module that is not supported by C++ API yet

class TestCppApiParity(common.TestCase):
  pass

parity_table_path = os.path.join(os.path.dirname(__file__), 'cpp_api_parity/parity-tracker.md')

parity_table = parse_parity_tracker_table(parity_table_path)

# yf225 TODO comment:
# RHS value format: 'input' / 'target' / 'extra_args_0' / 'extra_args_1'
# NOTE: any var symbol written in the cpp_* fields needs to have a mapping here!

for test_params_dicts, test_instance_class in [
  (sample_module.module_tests, common_nn.ModuleTest),
  (sample_functional.functional_tests, common_nn.NewModuleTest),
  (common_nn.module_tests, common_nn.ModuleTest),
  (common_nn.new_module_tests, common_nn.NewModuleTest),
  (common_nn.criterion_tests, common_nn.CriterionTest),
  (common_nn.new_criterion_tests, common_nn.NewCriterionTest),
]:
  module_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table, devices)
  functional_impl_check.add_tests(TestCppApiParity, test_params_dicts, test_instance_class, parity_table, devices)

# Assert that there exists auto-generated tests for `SampleModule` and `sample_functional`.
assert len([name for name in TestCppApiParity.__dict__ if 'SampleModule' in name]) == \
  len(sample_module.module_tests) * len(devices)
assert len([name for name in TestCppApiParity.__dict__ if 'sample_functional' in name]) == \
  len(sample_functional.functional_tests) * len(['cpu', 'cuda'])

module_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=print_cpp_source)
functional_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=print_cpp_source)

if __name__ == "__main__":
  common.run_tests()
