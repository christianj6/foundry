import os
import pandas as pd
from .objects import keywords, document
from .tools import get_distribution, evaluate_classifiers
from .train import train_and_save_new_model
from tqdm import tqdm


def run(**kwargs):
    if kwargs["train"] is True:
        assert (
            kwargs["keywords"] is not None
        ), "You must provide a keywords file when training."
        assert os.path.exists(
            kwargs["keywords"]
        ), "The keywords path must be a valid absolute path."
        assert kwargs["data"] is not None, "You must provide training data."
        assert os.path.exists(
            kwargs["data"]
        ), "The training data path must be a valid absolute path."

        train_and_save_new_model(
            kwargs["data"], kwargs["keywords"], previous_model=kwargs["model"]
        )

    elif kwargs["evaluate"] is True:
        assert (
            kwargs["keywords"] is not None
        ), "You must provide a keywords file when training."
        assert os.path.exists(
            kwargs["keywords"]
        ), "The keywords path must be a valid absolute path."
        assert kwargs["data"] is not None, "You must provide training data."
        assert os.path.exists(
            kwargs["data"]
        ), "The training data path must be a valid absolute path."
        assert (
            kwargs["model"] is not None
        ), "You must provide a model for evaluation."
        assert os.path.exists(
            kwargs["model"]
        ), "The model path must be a valid absolute path."

        print(
            evaluate_classifiers(
                kwargs["model"], kwargs["keywords"], kwargs["data"]
            )
        )

    else:
        assert all(
            [
                os.path.exists(pth)
                for pth in tuple(
                    map(lambda x: kwargs[x], ("keywords", "corpus"))
                )
            ]
        ), "You must provide a correct absolute path for both the keywords and corpus files."
        bound = 0.95
        if kwargs["bound"] is not None:
            bound = kwargs["bound"]

        trained_filter = False if kwargs["model"] is None else kwargs["model"]

        with open(kwargs["keywords"], "r") as f:
            words = f.read().splitlines()

        kw = keywords.Keywords(words=words, ids=list(range(len(words))))
        corpus = pd.read_csv(kwargs["corpus"])

        output = []
        for t in tqdm(corpus.text.tolist()[:10]):
            doc = document.Doc(
                text=t.split(" "),
                keywords=kw,
                file=kwargs["corpus"],
                language=kwargs["language"],
                bound=bound,
                trained_filter=trained_filter,
            )
            output.append(doc.entities)

        output = pd.concat(output)

        assert len(output) > 0, "No keywords were found."

        # Combine output and distribution into a single .xls
        writer = pd.ExcelWriter("output.xlsx")
        output.to_excel(writer, "keywords", index=False)
        # Get distribution for each file.
        for name, file in output.groupby(["Keyword"]):
            distribution = get_distribution(file)
            # Add as a new sheet.
            distribution.to_excel(writer, name, index=False)

        writer.save()
