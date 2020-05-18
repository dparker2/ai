import os
import pickle

import agents


MODELS_DIR = "models"


def model_filename(env: str, agent: str):
    return f"{env}-{agent}.pickle"


def save_model(env: str, agent: str, model: agents.Agent):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    location = os.path.join(MODELS_DIR, model_filename(env, agent))
    with open(location, "wb") as f:
        pickle.dump(model, f)


def load_model(env: str, agent: str) -> agents.Agent:
    location = os.path.join(MODELS_DIR, model_filename(env, agent))

    if not os.path.exists(location):
        return

    with open(location, "rb") as f:
        return pickle.load(f)
