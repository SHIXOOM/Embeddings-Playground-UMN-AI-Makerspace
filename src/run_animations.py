import json
import subprocess
from typing import List, Tuple
import numpy as np
import os

def run_animations(
    inputs: List[Tuple[str, np.ndarray]],
    ops: List[str],
    result: Tuple[str, np.ndarray],
    similars: List[Tuple[str, np.ndarray, float]]
):
    """
    Launch Manim animations for vector operations and similarity comparison.
    
    Args:
        inputs: List of (word, vector) pairs representing input words
        ops: List of operations like ["add", "sub"]
        result: Tuple of (result_word, result_vector)
        similars: List of (word, vector, similarity_score) for top similar words
    """
    # Prepare data for JSON file
    data = {
        "inputs":  [[w, v.tolist()] for w, v in inputs],
        "ops":     ops,
        "result":  [result[0], result[1].tolist()],
        "similars":[[w, v.tolist(), s] for w, v, s in similars],
    }
    
    # Write the data file beside animation.py
    cfg_path = os.path.join("src", "animation_data.json")
    with open(cfg_path, "w") as f:
        json.dump(data, f)

    # Now invoke manim with no custom flags needed
    cmd = [
        "manim", "-pql",
        "src/animation.py",
        "VectorOpsScene",
        "SimilarityScene"
    ]
    
    try:
        # this will render VectorOpsScene, then SimilarityScene, in one subprocess
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running animations: {e}")
    except FileNotFoundError:
        print("Error: Manim not found. Please install manim: pip install manim")
