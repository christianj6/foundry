import pandas as pd
from .objects import keywords
import os
import dill
import re


def json_to_str(json: dict, ret: str, txt: str) -> str:
    """
    Concatenates text attributes from
    json files into a single, string
    of text.

    Parameters
    ---------
        json : dict
            Loaded json dictionary.
        ret : str
            Attribute for return value.
        txt : str
            Attribute for text value.

    Returns
    ---------
        text : list[str]
            List of page texts as cleaned
            strings.
    """
    text = []
    for page in json[ret]:
        text.append(page[txt].strip())

    return [" ".join(item.split()) for item in text]


def get_distribution(output: "pd.DataFrame"):
    """
    Get keyword distribution statistics
    that we can output these
    to the final csv.

    Parameters
    ---------
        output : pd.DataFrame
            Output with
            identified entities.

    Returns
    ---------
        distribution : pd.DataFrame
            Distribution statistics
            in tabular form.
    """
    distribution = []
    for name, group in output.groupby(["Keyword"]):
        distribution.append(
            {
                "Keyword": name,
                "Count": len(group),
            }
        )

    return pd.DataFrame(distribution)


def evaluate_classifiers(model_path, keywords_file, data_path):
    """
    Evaluate classification
    accuracy in a messy fashion,
    by averaging all results
    to see how the classifers
    are collectively performing.

    Returns
    ---------
        score : float
            Classification accuracy.
    """
    with open(keywords_file, "r") as f:
        words = f.read().splitlines()

    kw = keywords.Keywords(words=words, ids=list(range(len(words))))

    scores = []
    output = pd.read_excel(data_path).infer_objects()

    for file in os.listdir(model_path):
        with open(f"{model_path}/{file}", "rb") as f:
            model = dill.load(f)

        rows = output[output["Keyword"] == file.strip(".pb")]
        X = rows["Surrounding Text"].tolist()
        X = [re.sub(r"[A-Z]", "", str(x)) for x in X]
        X = [kw.get_vector(x).toarray()[0] for x in X]
        y = rows["Match is Invalid"].tolist()
        scores.append((file, model.model.score(X, y)))

    return scores
