import itertools
import json
import os
from typing import List, Tuple
import csv
import pandas as pd
import sys
import numpy as np

def merge(dict1, dict2):
    """
    Merge two bounds dictionaries recursively
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2

    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k] = merge(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]

    return dict1

class RuleGenerator:
    def __init__(self, schema_file: str) -> None:
        self.schema_file: str = schema_file
        self.__load_schema()
        self.__assert_schema()

    def __load_schema(self) -> None:
        with open(self.schema_file, 'r') as f:
            self.schema: dict = json.load(f)

    def __assert_schema(self) -> None:
        assert 'classes' in self.schema.keys(), f'KeyError: Schema has no classes'
        assert 'stats' in self.schema.keys(), f'KeyError: Schema has no statistics'
        assert 'form' in self.schema.keys(), f'KeyError: Schema has no form'
        assert 'delta' in self.schema.keys(), f'KeyError: Schema has no delta'
        assert 'input-file' in self.schema.keys(), f'KeyError: Schema has no input file'
        assert self.schema['form'] in ('two-sided', 'logic'), f'ValueError: Schema form must be either two-sided or logic'
        if 'output-format' not in self.schema.keys():
            self.schema['output-format'] = 'json'
        # if self.schema['form'] == 'logic':
        #     assert 'modes' in self.schema.keys(), f'KeyError: Schema has no modes'

    def calculate_bounds(self, values: List[float], delta: int = 98) -> Tuple[float, float]:
        threshold = 100 - ((100 - delta) / 2)
        
        sorted(values)

        if len(values) == 0:
            return ()

        npperc01 = np.percentile(values, 100 - threshold)
        npperc99 = np.percentile(values, threshold)

        lower_bound = npperc01
        upper_bound = npperc99

        conforming_f1s = [ element for element in values if element >= lower_bound and element <= upper_bound ]

        return (lower_bound, upper_bound), len(conforming_f1s) / len(values)

    def generate_abstract_rules(self) -> list:
        pass


class TwoSidedRuleGenerator(RuleGenerator):
    def __init__(self, schema_file: str) -> None:
        super().__init__(schema_file)
        assert self.schema['form'] == 'two-sided', f'ValueError: Schema form must be two-sided'

    def generate_abstract_rules(self) -> list:
        nesting_level = 2
        # ILASP command: ./ILASP --version=3 -nc -na  -s ~/research/neural_verification_exploration/imagenetx/commonsense_stuff/modes.txt | tee commonsense_imagenet_sspace.txt

        # generating combinations for statistics
        perms = []
        for i in range(1, nesting_level + 1):
            perms += list(itertools.permutations(self.schema['stats'], i))

        return perms
    
    def get_bounds(self, rule: tuple, df: pd.DataFrame) -> dict:
        """
        Given a rule, calculate the bounds for each statistic
        """
        bounds = {}
        if len(rule) == 1:
            bounds[rule[0]] = self.calculate_bounds(list(df[rule[0]]), self.schema['delta'])
        else:
            # get quartiles for first statistic, then recursively call for the rest
            quartile_stat = rule[0]
            quartiles = np.quantile(df[quartile_stat], [0.25, 0.5, 0.75])
            bounds = { f"{rule[0]}_groups" : {
                "q1": {
                    "ubound": quartiles[0],
                    "conf_bounds": self.get_bounds(rule[1:], df[df[quartile_stat] <= quartiles[0]])
                },
                "q2": {
                    "lbound": quartiles[0],
                    "ubound": quartiles[1],
                    "conf_bounds": self.get_bounds(rule[1:], df[(df[quartile_stat] > quartiles[0]) & (df[quartile_stat] <= quartiles[1])])
                },
                "q3": {
                    "lbound": quartiles[1],
                    "ubound": quartiles[2],
                    "conf_bounds": self.get_bounds(rule[1:], df[(df[quartile_stat] > quartiles[1]) & (df[quartile_stat] <= quartiles[2])])
                },
                "q4": {
                    "lbound": quartiles[2],
                    "conf_bounds": self.get_bounds(rule[1:], df[df[quartile_stat] > quartiles[2]])
                },
            }}

        return bounds
    
    def generate_concrete_rules(self, abstract_rules: List[tuple]) -> dict:
        """
        First load the data
        Data format:
        label, stat1, stat2, stat3, ...
        
        Then, collate the data as per the abstract rule requirements
        """

        stats_data = pd.read_csv(self.schema['input-file'], sep='\t')

        assert all([ stat in stats_data.columns for stat in self.schema['stats'] ]), f'ValueError: Statistics not found in data'
        assert 'class' in stats_data.columns, f'ValueError: Class not found in data'

        concrete_rules = {}

        for cls in self.schema['classes']:
            concrete_rules[cls] = {}
            for rule in abstract_rules:
                new_rules = self.get_bounds(rule, stats_data[stats_data['class'] == cls])
                concrete_rules[cls] = merge(concrete_rules[cls], new_rules)

        return concrete_rules
    
    def write_rules(self, concrete_rules: dict) -> None:
        output_file = self.schema['output-file']
        if output_file.endswith('json'):
            with open(output_file, 'w') as f:
                json.dump(concrete_rules, f, indent=4)
        else:
            assert output_file.endswith('csv'), f'ValueError: Output file must be either a json or a csv file'
            # SUPPORT ONLY FOR RULES WITH CLASSES NESTED AT MOST 2 LEVELS

            rows = []
            for cls, stats in concrete_rules.items():
                assert cls in self.schema['classes'], f'ValueError: Class {cls} not found in schema'
                for statname, stat_bound in stats.items():
                    if not statname.endswith('_groups'):
                        rows.append([('label', cls, cls), (statname, stat_bound[0][0], stat_bound[0][1])])
                    else:
                        grp_stat = statname.split('_groups')[0]
                        for gid, group in stat_bound.items():
                            assert gid.startswith('q')
                            gub = group['ubound'] if 'ubound' in group.keys() else np.inf
                            glb = group['lbound'] if 'lbound' in group.keys() else -np.inf
                            group_filters = [('label', cls, cls), (grp_stat, glb, gub)]
                            for stat, bound in group['conf_bounds'].items():
                                rows.append(group_filters + [(stat, bound[0][0], bound[0][1])])

            with open(output_file, 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                for row in rows:
                    writer.writerow(row)


class LogicRuleGenerator(RuleGenerator):
    # WIP
    def __init__(self, schema_file: str) -> None:
        super().__init__(schema_file)
        assert self.schema['form'] == 'logic', f'ValueError: Schema form must be logic'
        assert 'modes-file' in self.schema.keys(), f'KeyError: Schema has no modes file'

    def generate_abstract_rules(self) -> list:
        rules = []
        modes_file = self.schema['modes-file']
        # uses ILASP-3.1.0

        command = f'./ILASP --version=3 -nc -na -s {modes_file} | tee abstract_rules.txt'
        os.system(command)

        with open('abstract_rules.txt', 'r') as f:
            for line in f.readlines():
                rule = line.strip().split(' ~ ')[1]
                rule = rule.replace('not ', '!')
                rules.append(rule)
        
        return rules
    

if __name__ == "__main__":
    schema_file = 'schema.json'
    rulegen = TwoSidedRuleGenerator()
    abstract_rules = rulegen.generate_abstract_rules()
    concrete_rules = rulegen.generate_concrete_rules(abstract_rules)
    rulegen.write_rules(concrete_rules)
