# Adapted from: https://github.com/binghong-ml/retro_star

import os
import numpy as np
import logging
import time
from .mol_tree import MolTree

def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None, max_time=300):
    
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn
    )

    i = -1
    start_time = time.time()

    if not mol_tree.succ:
        for i in range(iterations):
            if time.time() - start_time > max_time:
                break
            
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            result = expand_fn(m_next.mol)

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                analysis_tokens = result['analysis']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                templates = result['templates']
                
                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                succ = mol_tree.expand(m_next, reactant_lists, costs, templates, analysis_tokens)

                if succ:
                    break

                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    break

            else:
                mol_tree.expand(m_next, None, None, None, None)

        search_time = time.time() - start_time

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    return mol_tree.succ, best_route, i+1