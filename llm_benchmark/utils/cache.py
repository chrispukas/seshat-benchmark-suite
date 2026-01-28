import os

import pickle as pkl

from typing import List, Tuple, Dict, Optional


def exists_file(filepath: str
                ) -> bool:
    return os.path.isfile(filepath)

def save_file(data: object, 
              filepath: str,
              overwrite: Optional[bool] = True) -> None:
    if not overwrite and os.path.isfile(filepath):
        print("Overwrite disabled.")
        return
    
    try:
        pkl.dump(obj=data, 
                 file=open(filepath, "wb")
                 )
        print(f"Saved pickle to {filepath}.")
    except pkl.PicklingError:
        print(f"Failed to save pickle {filepath}.")

def load_file(filepath: str
              ) -> object:
    
    if not os.path.isfile(filepath):
        print(f"File {filepath} not found.")
        return None

    try:
        return pkl.load(file=open(filepath, "rb"))
    except pkl.UnpicklingError:
        print(f"Failed to load pickle {filepath}.")