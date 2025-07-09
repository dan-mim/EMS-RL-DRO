from pystocoptim import Atom
import numpy as np
import random
import networkx as nx


__all__ = ["filtration_into_networkx_tree", "generate_uniform_tree"]


def generate_uniform_tree(typearbre=2, seed=0, depth=5):
    seed_value = seed
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Initialisation d'une filtration (liste de listes d'atomes)
    filtration = []

    # Création des atomes par niveaux (profondeur 0 à depth)
    atom_id = 0

    # Construction des niveaux de la filtration
    for d in range(depth + 1):
        level = []
        for _ in range(typearbre ** d):  # Nombre d'atomes = typearbre^d pour chaque niveau
            atom = Atom(scen_ids={atom_id}, depth=d, id=atom_id)
            level.append(atom)
            atom_id += 1
        filtration.append(level)

    # Génération des probabilités des feuilles (dernier niveau)
    leaves = filtration[-1]
    probabilities = np.random.dirichlet(np.ones(len(leaves)))  # Somme des probabilités des feuilles = 1

    # Attribution des probabilités et des valeurs aux feuilles
    for i, leaf in enumerate(leaves):
        leaf.proba = probabilities[i]
        leaf.value = np.random.uniform(-10, 10, (1,))

    # Propagation des probabilités et des valeurs vers les parents
    for d in range(depth - 1, -1, -1):
        level = filtration[d]
        next_level = filtration[d + 1]

        for i, atom in enumerate(level):
            children = [next_level[typearbre * i + j] for j in range(typearbre)]  # Récupère 'typearbre' enfants
            atom.children = children

            # Définir les parents pour les enfants
            for child in children:
                child.parent = atom

            # Somme des probabilités des enfants pour obtenir celle du parent
            total_proba = sum(child.proba for child in children)
            atom.proba = total_proba

            # Moyenne des valeurs des enfants pour obtenir celle du parent
            # avg_value = sum(child.value() for child in children) / typearbre
            atom.value = np.random.uniform(-10, 10, (1,)) #.set_value(avg_value)

    return filtration


def filtration_into_networkx_tree(filtration):
    G = nx.DiGraph()  # Initialize a directed graph

    # Loop over all levels of the filtration
    for level in filtration:
        for atom in level:
            # Add node with attributes 'quantizer' (value) and 'stage' (depth)
            G.add_node(atom.id, quantizer=atom.value, stage=atom.depth)

            # If the atom has children, add directed edges with 'weight' (conditional probability)
            if atom.children:
                for child in atom.children:
                    parent_proba = atom.proba
                    child_proba = child.proba

                    # Calculate the conditional probability
                    if parent_proba > 0:
                        conditional_proba = child_proba / parent_proba
                    else:
                        conditional_proba = 0  # Avoid division by zero if the parent probability is zero

                    # Add a directed edge from parent to child with the conditional probability as weight
                    G.add_edge(atom.id, child.id, weight=conditional_proba)

    return G


def compute_filtration_structure(seed=0, depth=145, fan_lenght=6):
    seed_value = seed
    np.random.seed(seed_value)
    random.seed(seed_value)

    # Initialisation d'une filtration (liste de listes d'atomes)
    filtration = []

    # Création des atomes par niveaux (profondeur 0 à depth)
    atom_id = 0

    # Construction des niveaux de la filtration
    nb_atoms = 1
    for d in range(depth + 1):
        level = []
        if d%fan_lenght==0 and d>0:
            nb_atoms = nb_atoms * 2
        for _ in range(nb_atoms):  # Nombre d'atomes = typearbre^nb_atoms pour chaque niveau
            atom = Atom(scen_ids={atom_id}, depth=d, id=atom_id)
            level.append(atom)
            atom_id += 1
        filtration.append(level)

    # Génération des probabilités des feuilles (dernier niveau)
    leaves = filtration[-1]
    probabilities = np.random.dirichlet(np.ones(len(leaves)))  # Somme des probabilités des feuilles = 1

    # Attribution des probabilités et des valeurs aux feuilles
    for i, leaf in enumerate(leaves):
        leaf.proba = probabilities[i]
        leaf.value = np.random.uniform(-10, 10, (1,))

    # Propagation des probabilités et des valeurs vers les parents
    for d in range(depth - 1, -1, -1):
        level = filtration[d]
        next_level = filtration[d + 1]

        if len(level) == len(next_level):
            for i, atom in enumerate(level):
                children = [next_level[i]]
                atom.children = children

                # Définir les parents pour les enfants
                for child in children:
                    child.parent = atom

                # Somme des probabilités des enfants pour obtenir celle du parent
                total_proba = sum(child.proba for child in children)
                atom.proba = total_proba

                # Moyenne des valeurs des enfants pour obtenir celle du parent
                # avg_value = sum(child.value() for child in children) / typearbre
                atom.value = np.random.uniform(-10, 10, (1,)) #.set_value(avg_value)

        else:
            s = 0
            for i, atom in enumerate(level):
                s = i * 2
                children = next_level[i*2:i*2+2]  # Récupère 'typearbre' enfants
                atom.children = children

                # Définir les parents pour les enfants
                for child in children:
                    child.parent = atom

                # Somme des probabilités des enfants pour obtenir celle du parent
                total_proba = sum(child.proba for child in children)
                atom.proba = total_proba

                # Moyenne des valeurs des enfants pour obtenir celle du parent
                # avg_value = sum(child.value() for child in children) / typearbre
                atom.value = np.random.uniform(-10, 10, (1,))  # .set_value(avg_value)
    return filtration