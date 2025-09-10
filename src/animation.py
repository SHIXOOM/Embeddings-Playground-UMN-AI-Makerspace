import json
import os
from manim import Scene, Arrow, Text, VGroup, MathTex, Write, Create, FadeIn, FadeOut, ReplacementTransform, UP, DOWN, RIGHT
from typing import List, Tuple
import numpy as np

# --- load the configuration ---
_cfg_path = os.path.join(os.path.dirname(__file__), "animation_data.json")
try:
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
    
    # unpack
    _inputs  = [(w, np.array(v)) for w, v      in _cfg["inputs"]]
    _ops     = _cfg["ops"]
    _result  = (_cfg["result"][0], np.array(_cfg["result"][1]))
    _similars= [(w, np.array(v), s) for w, v, s in _cfg["similars"]]
except FileNotFoundError:
    # Default data for testing if JSON file doesn't exist
    _inputs = [("king", np.array([1.0, 0.5, 0.3])), ("man", np.array([0.9, 0.45, 0.25])), ("woman", np.array([1.1, 0.55, 0.35]))]
    _ops = ["sub", "add"]
    _result = ("queen", np.array([1.2, 0.6, 0.45]))
    _similars = [("princess", np.array([1.15, 0.58, 0.42]), 0.99), ("monarchy", np.array([1.05, 0.52, 0.38]), 0.96)]

class VectorOpsScene(Scene):
    """
    A Manim scene to visualize vector operations (addition and subtraction) on word embeddings.
    """
    def construct(self):
        origin = np.array([-4, 0, 0])
        
        # Reduce to 2D for visualization and pad with zeros for 3D compatibility
        vectors = [np.append(v[:2], 0) for _, v in _inputs]
        words = [w for w, _ in _inputs]
        
        # Display initial vectors
        arrows = VGroup(*[Arrow(origin, origin + vec, buff=0) for vec in vectors])
        labels = VGroup(*[Text(word).next_to(arrow.get_end(), UP) for word, arrow in zip(words, arrows)])
        
        self.play(Create(arrows), Write(labels))
        self.wait(1)

        result_vec = vectors[0].copy()
        result_arrow = arrows[0].copy()

        for i, op in enumerate(_ops):
            op_symbol = MathTex("+").scale(2) if op == "add" else MathTex("-").scale(2)
            next_vec = vectors[i+1]
            next_arrow = arrows[i+1]
            
            # Place operation symbol between vectors
            midpoint = (result_arrow.get_end() + next_arrow.get_end()) / 2
            op_symbol.move_to(midpoint)
            
            self.play(FadeIn(op_symbol))
            self.wait(0.5)

            if op == "add":
                new_result_vec = result_vec + next_vec
            else: # subtract
                new_result_vec = result_vec - next_vec

            new_arrow = Arrow(origin, origin + new_result_vec, buff=0, color=result_arrow.get_color())
            
            # Show the operation by moving the second vector
            self.play(
                next_arrow.animate.shift(result_arrow.get_end() - origin),
            )
            self.wait(0.5)
            
            self.play(
                ReplacementTransform(VGroup(result_arrow, next_arrow), new_arrow),
                FadeOut(op_symbol)
            )
            result_arrow = new_arrow
            result_vec = new_result_vec
            self.wait(1)

        final_label = Text("Result").next_to(result_arrow.get_end(), UP)
        self.play(Write(final_label))
        self.wait(2)


class SimilarityScene(Scene):
    """
    A Manim scene to visualize cosine similarity between a result vector and similar word vectors.
    """
    def construct(self):
        # Unpack result and similars
        result_word, result_vector = _result
        result_vector = np.append(result_vector[:2], 0)  # Reduce to 2D and pad with 0 for 3D compatibility
        similars = [(w, np.append(v[:2], 0), s) for w, v, s in _similars]

        # Result vector on the left
        result_origin = np.array([-5, 0, 0])
        result_arrow = Arrow(result_origin, result_origin + result_vector, buff=0, color="#FFD700")
        result_label = Text(result_word).next_to(result_arrow.get_end(), UP)
        
        self.play(Create(result_arrow), Write(result_label))
        self.wait(1)

        # Similar vectors on the right
        similar_origin = np.array([2, 2.5, 0])
        similar_group = VGroup()
        for i, (word, vec, score) in enumerate(similars):
            arrow = Arrow(similar_origin, similar_origin + vec, buff=0)
            label = Text(word).next_to(arrow.get_end(), RIGHT)
            entry = VGroup(arrow, label).shift(DOWN * i * 2)
            similar_group.add(entry)

        self.play(Create(similar_group))
        self.wait(1)

        # Cosine similarity calculation
        formula = MathTex(r"\cos\theta = \frac{A \cdot B}{\|A\| \|B\|}").to_edge(UP)
        self.play(Write(formula))
        self.wait(1)

        for i, (word, vec, score) in enumerate(similars):
            sim_arrow = similar_group[i][0]
            
            # Highlight vectors being compared
            self.play(
                result_arrow.animate.set_color("#FF4500"),
                sim_arrow.animate.set_color("#00BFFF")
            )

            # Show score
            score_text = MathTex(f"\\approx {score:.2f}").next_to(sim_arrow, DOWN)
            self.play(Write(score_text))
            self.wait(1)

            # Reset colors
            self.play(
                result_arrow.animate.set_color("#FFD700"),
                sim_arrow.animate.set_color("#FFFFFF"),
                FadeOut(score_text)
            )
        
        self.wait(2)
