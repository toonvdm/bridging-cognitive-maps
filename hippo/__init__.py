from pathlib import Path
from datetime import datetime
import yaml
import pickle

import __main__ as main


class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        remap = {"hippo.models.cscg_untouched": "hippo.models.cscg"}
        module = remap.get(module, module)
        return super().find_class(module, name)


def save_object(obj, store_path):
    with open(store_path, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(load_path):
    with open(load_path, "rb") as inp:
        agent = RenamingUnpickler(inp).load()
    return agent


def save_config(config, store_path):
    with open(store_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def load_config(load_path):
    with open(load_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def is_interactive():
    return not hasattr(main, "__file__")


def get_data_path():
    p = Path(__file__).parent / "../data"
    return p.resolve()


def get_recents_path():
    p = get_data_path() / "working/recent"
    p.mkdir(exist_ok=True, parents=True)
    return p


def get_store_path(name):
    # Group per day
    day = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H-%M-%S")
    dp = get_data_path() / "working" / name / day / now
    dp.mkdir(exist_ok=True, parents=True)
    return dp.resolve()
