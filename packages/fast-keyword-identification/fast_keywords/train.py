from .objects import keywords, trainer
import pandas as pd
import dill
import re
import os


def train_and_save_new_model(
    training_data, keywords_file, previous_model=None
):
    """
    Trains and saves a new model to the current working directory.
    """
    with open(keywords_file, "r") as f:
        words = f.read().splitlines()

    kw = keywords.Keywords(words=words, ids=list(range(len(words))))
    os.makedirs(f"{os.getcwd()}/model/", exist_ok=True)
    try:
        output = pd.read_excel(training_data).infer_objects()

    except Exception:
        raise FileNotFoundError

    for name, group in output.groupby(["Keyword"]):
        print(f"Training model for: {name}")
        try:
            if group["Match is Invalid"].astype("int32").any():
                environment_vectors = []
                environment_vector_labels = []
                for i, (_, row) in enumerate(group.iterrows()):
                    environment = row["Surrounding Text"]
                    # Remove capital letters ie the original entity.
                    environment = re.sub(r"[A-Z]", "", str(environment))
                    environment_vectors.append(
                        kw.get_vector(environment).toarray()[0]
                    )
                    environment_vector_labels.append(
                        int(row["Match is Invalid"])
                    )

                try:
                    # Try to load a pre-existing model.
                    with open(f"{previous_model}/{name}", "rb") as f:
                        model = dill.load(f)

                    # Append data and labels to model.
                    model.data.extend(environment_vectors)
                    model.labels.extend(environment_vector_labels)

                except FileNotFoundError:
                    # If does not exist, create new trainer object.
                    model = trainer.Trainer(
                        keyword=name,
                        data=environment_vectors,
                        labels=environment_vector_labels,
                    )

                # Fit the trainer object to the updated data for that word.
                model.train()
                # Save the fitted object to models dir for use during runtime.
                with open(f"{os.getcwd()}/model/{name}.pb", "wb") as f:
                    dill.dump(model, f)

        except ValueError:
            pass
