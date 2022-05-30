import pandas as pd
import collections


def create_ancestors_dict(path_to_ancestors, path_to_weights=None):
    """_summary_

    Args:
        path_to_ancestors (string): path to the ancestors file
        path_to_weights (string, optional): path to the weight file. Defaults to None, uniform.

    Returns:
        dict: ancestors dict d[MESH][ancestors] for the ancestors list and d[MESH][proba] for the associated probabilities
    """

    ancestors_dict = collections.defaultdict(dict)

    # From ancestors csv file to dict
    ancestors = pd.read_csv(path_to_ancestors)
    pre_ancestors_dict = ancestors.groupby('MESH')['MESH_ANCESTOR'].apply(list).to_dict()

    # Create weight dict
    weights = dict()
    if path_to_weights:
        weights = pd.read_csv(path_to_weights).groupby("MESH")["TOTAL_PMID_MESH"].apply(int).to_dict()

    # browse ancestors and insert weights
    for mesh, ancestors in pre_ancestors_dict.items():

        # Add the MeSH itselft to the list of ancestors
        ancestors_dict[mesh]["ancestors"] = ancestors
        inverse_weights = [1/(weights.get(ancestor, 0) + 1) for ancestor in ancestors]
        ancestors_dict[mesh]["proba"] = [w/sum(inverse_weights) for w in inverse_weights]

    return ancestors_dict


a = create_ancestors_dict("data/mesh_ancestors.csv", "data/mesh_pmids_count.csv")