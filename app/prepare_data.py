import pandas as pd
import collections
import requests
import sys
import random
import csv
from io import StringIO


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



def extract_samples(url, n, max_int, tree):

    # Estimate steps, limits and offset for sparql requests
    n_steps = n // max_int
    rest = n % max_int
    offset_list = [i * max_int for i in range(n_steps)]
    limit_list = [max_int] * n_steps
    if (rest):
        offset_list.append(n_steps * max_int)
        limit_list.append(rest)

    # Request parameters
    header = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/csv"
    }

    data = {
        "format": "csv",
    }

    request = """
    DEFINE input:inference "schema-inference-rules"

    select distinct (strafter(str(?pmid), "http://rdf.ncbi.nlm.nih.gov/pubchem/reference/PMID") as ?PMID) (strafter(STR(?mesh),"http://id.nlm.nih.gov/mesh/") as ?MESH)
    where
    {
        {
            select ?pmid
            where
            {
                {
                    select distinct ?pmid
                    where
                    {
                        ?pmid a fabio:Article .
                        ?pmid dcterms:date ?date
                    }
                    ORDER BY DESC(?date)
                }
            }
            LIMIT %(limit)s
            OFFSET %(offset)s
        }
    ?pmid (fabio:hasSubjectTerm|fabio:hasSubjectTerm/meshv:hasDescriptor) ?mesh .
    ?mesh a meshv:TopicalDescriptor .
    ?mesh meshv:active 1 .
    ?mesh meshv:treeNumber ?tn .
    FILTER(REGEX(?tn,"%(tree)s")) .
    }

"""

    dataset = pd.DataFrame()

    for i in range(len(offset_list)):

        # Send request
        offset, limit = offset_list[i], limit_list[i]
        print(f"Offset number {offset} with limit: {limit}")
        filled_request = request % {"offset" : offset, "limit" : limit, "tree": tree}
        data["query"] = filled_request
        print(filled_request)
        r = requests.post(url=url, headers=header, data=data)

        # Check request status
        if r.status_code != 200:
            print("Error in request")
            sys.exit(3)

        # Convert string to dataframe
        csvStringIO = StringIO(r.text)
        df = pd.read_csv(csvStringIO, sep=",")
        print(len(set(df["PMID"])))

        # Concat dataframe to dataset
        dataset = pd.concat([dataset, df])
    
    return dataset


def create_samples(n, data, w, ancestors, f_input, f_target):
    """
    n(int): nombre d'exemples à générer
    data(fichier format CSV): les pmids avec les mesh annotés
    w(int): taille de la fenêtre, nombre de meshs pour le contexte
    ancestors(dico {mesh:poids}): ancêtres du mesh dans arbre
    f_input(chemin fichier CSV): fichier où enregistrer le contexte
    f_target(chemin fichier CSV): fichier où enregistrer les target
    """
    
    context = []
    target = []
    
    df = pd.read_csv(data)
    pmids = [id for id in df["PMID"].unique()]

    i = 0
    while i < n:
        pmid = random.choice(pmids)
        mesh = df[df["PMID"] == pmid]["MESH"].to_list()

        # target = 1 mesh au hasard
        mesh_target = mesh.pop(random.randrange(len(mesh)))
        mesh_target = random.choices(ancestors[mesh_target]["ancestors"], weights=ancestors[mesh_target]["proba"], k=1)
        target.append(mesh_target)

        # contexte = w mesh au hasard
        if w > len(mesh):
            # supprimer
            target.pop()
        else:
            mesh_context = [mesh.pop(random.randrange(len(mesh))) for j in range(w)]
            mesh_context = [random.choices(ancestors[m]["ancestors"], weights=ancestors[m]["proba"], k=1)[0] for m in mesh_context]
            context.append(mesh_context)
            i += 1

    with open(f_input, "w") as f:
        write = csv.writer(f)
        write.writerows(context)
    with open(f_target, "w") as f:
        write = csv.writer(f)
        write.writerows(target)


# sampling_100000_pmids_mesh = extract_samples(url="https://forum.semantic-metabolomics.fr/sparql", n=100000, max_int=10000, tree="(C|A|G|F|I|J|D20|D23|D26|D27)")
# sampling_100000_pmids_mesh.to_csv("data/sampling_100000_pmids_mesh.csv", header=True, index=False)



ancestors_dict = create_ancestors_dict("data/mesh_ancestors.csv", "data/mesh_pmids_count.csv")
print(ancestors_dict["D010084"])

create_samples(1000000, "data/sample_1000000/sampling_100000_pmids_mesh.csv", 6, ancestors_dict, "data/sample_1000000/context.csv", "data/sample_1000000/target.csv")