# Adapted from: https://github.com/binghong-ml/retro_star

import numpy as np
from queue import Queue

class SynRoute:
    def __init__(self, target_mol, succ_value, search_status):
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.values = [None]
        self.templates = [None]
        self.parents = [-1]
        self.children = [None]
        self.optimal = False
        self.costs = {}
        self.analysis_dict = {}

        self.succ_value = succ_value
        self.total_cost = 0
        self.length = 0
        self.search_status = search_status
        if self.succ_value <= self.search_status:
            self.optimal = True

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.values.append(None)
        self.templates.append(None)
        self.parents.append(parent_id)
        self.children.append(None)

        self.children[parent_id].append(len(self.mols) - 1)

    def set_value(self, mol, value):
        assert mol in self.mols

        mol_id = self.mols.index(mol)
        self.values[mol_id] = value

    def add_reaction(self, mol, value, template, analysis_tokens, reactants, cost):
        assert mol in self.mols

        self.total_cost += cost
        self.length += 1

        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.templates[parent_id] = template
        self.children[parent_id] = []
        self.costs[parent_id] = cost
        self.analysis_dict[parent_id] = analysis_tokens

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def serialize_reaction(self, idx):
        s = self.mols[idx]
        if self.children[idx] is None:
            return s, 0.0
        s += ">>"
        cost = np.exp(-self.costs[idx])
        analysis = self.analysis_dict[idx]
        template = self.templates[idx]
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += "."
            s += self.mols[self.children[idx][i]]
        return s, cost, analysis, template

    def get_reaction_list(self):
        total_reactions, total_cost = [], []
        total_analysis = []
        total_templates = []
        reaction, cost, analysis, template = self.serialize_reaction(0)
        total_reactions.append(reaction)
        total_cost.append(cost)
        total_analysis.append(analysis)
        total_templates.append(template)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                reaction, cost, analysis, template = self.serialize_reaction(i)
                total_cost.append(cost)
                total_reactions.append(reaction)
                total_analysis.append(analysis)
                total_templates.append(template)
        return total_reactions, total_templates, total_cost, total_analysis
        
    def get_template_list(self):
        return self.templates