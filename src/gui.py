from tkinter import messagebox
from customtkinter import CTk, CTkLabel, CTkEntry, CTkButton, CTkCanvas, set_appearance_mode, set_default_color_theme, CTkFont
from gensim.models.keyedvectors import KeyedVectors
from embeddings_loader import load_static_embeddings
from run_animations import run_animations
import math
import tkinter as tk
import numpy as np

class WheelPicker(CTkCanvas):
    def __init__(self, master, items=None, radius_ratio=0.3, **kwargs):
        super().__init__(master, **kwargs)
        self.items = items or []
        self.radius_ratio = radius_ratio  # Radius as a ratio of canvas size
        self.angle = 0.0
        self.text_ids = []
        self.score_ids = []
        self._drag_start = None
        self._velocity = 0.0
        self._is_dragging = False
        self._selected_index = 0
        
        # Bind mouse events
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", self._on_configure)  # Bind resize event
        
        # Animation variables
        self._animation_id = None
        
        # Draw initial wheel
        self.after(100, self._draw_wheel)  # Delay to ensure canvas is properly sized

    def update_items(self, items):
        """Update the items displayed in the wheel"""
        self.items = items
        self.angle = 0.0
        self._selected_index = 0
        self._draw_wheel()

    def _on_configure(self, event):
        """Handle canvas resize events"""
        self._draw_wheel()

    @property
    def radius(self):
        """Calculate radius based on current canvas size"""
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return 80  # Default radius
        # Use ratio of the smaller dimension to ensure the circle fits
        min_dimension = min(width, height)
        return max(50, int(min_dimension * self.radius_ratio))

    def _draw_wheel(self):
        """Draw the wheel with items positioned in a circle"""
        # Clear previous items
        for tid in self.text_ids + self.score_ids:
            self.delete(tid)
        self.text_ids.clear()
        self.score_ids.clear()

        if not self.items:
            return

        # Get canvas dimensions
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:  # Canvas not ready yet
            self.after(50, self._draw_wheel)
            return
            
        cx = width // 2
        cy = height // 2
        
        n = len(self.items)
        if n == 0:
            return

        # Draw items around the circle
        for i, (word, score) in enumerate(self.items):
            theta = self.angle + 2 * math.pi * i / n
            x = cx + self.radius * math.sin(theta)
            y = cy - self.radius * math.cos(theta)
            
            # Calculate distance from top center to determine opacity and size
            distance_from_top = abs(theta % (2 * math.pi) - 0)
            if distance_from_top > math.pi:
                distance_from_top = 2 * math.pi - distance_from_top
            
            # Scale font size and opacity based on position
            scale = max(0.5, 1 - distance_from_top / math.pi)
            
            # Calculate adaptive font size based on canvas size and scale
            min_dimension = min(width, height)
            base_font_size = max(10, int(min_dimension / 20))  # Adaptive base font size
            font_size = int(base_font_size * scale)
            score_font_size = max(8, font_size - 4)
            
            # Calculate adaptive spacing based on font size
            word_offset = max(8, int(font_size * 0.6))  # Distance above center
            score_offset = max(8, int(score_font_size * 0.8))  # Distance below center
            
            # Determine color based on theme
            if scale > 0.8:  # Top item (selected)
                text_color = "#1f538d"  # Highlighted color
                self._selected_index = i
            else:
                text_color = "#7a7a7a"  # Dimmed color
            
            # Draw word
            tid = self.create_text(x, y - word_offset, text=word, 
                                 font=("Arial", font_size, "bold"), 
                                 fill=text_color, anchor="center")
            self.text_ids.append(tid)
            
            # Draw score below word
            sid = self.create_text(x, y + score_offset, text=f"{score}%", 
                                 font=("Arial", score_font_size), 
                                 fill=text_color, anchor="center")
            self.score_ids.append(sid)

    def _on_press(self, event):
        """Handle mouse press - start dragging"""
        self._drag_start = (event.x, event.y)
        self._is_dragging = True
        self._velocity = 0.0
        if self._animation_id:
            self.after_cancel(self._animation_id)

    def _on_drag(self, event):
        """Handle mouse drag - rotate the wheel"""
        if not self._is_dragging or not self._drag_start:
            return
            
        x0, y0 = self._drag_start
        dx = event.x - x0
        
        # Convert horizontal drag to angular rotation
        self.angle += dx / 100.0
        self._velocity = dx / 10.0  # Store velocity for momentum
        
        self._drag_start = (event.x, event.y)
        self._draw_wheel()

    def _on_release(self, event):
        """Handle mouse release - add momentum and snapping"""
        self._is_dragging = False
        self._add_momentum()

    def _add_momentum(self):
        """Add momentum after drag release and snap to nearest item"""
        if abs(self._velocity) > 0.1:
            self.angle += self._velocity
            self._velocity *= 0.95  # Friction
            self._draw_wheel()
            self._animation_id = self.after(16, self._add_momentum)  # ~60 FPS
        else:
            self._snap_to_nearest()

    def _snap_to_nearest(self):
        """Snap to the nearest item position"""
        if not self.items:
            return
            
        n = len(self.items)
        # Calculate the nearest "snap" position
        target_angle = round(self.angle * n / (2 * math.pi)) * (2 * math.pi) / n
        
        # Smooth animation to target
        diff = target_angle - self.angle
        if abs(diff) > 0.01:
            self.angle += diff * 0.2
            self._draw_wheel()
            self._animation_id = self.after(16, self._snap_to_nearest)

    def get_selected_item(self):
        """Get the currently selected item"""
        if not self.items or self._selected_index >= len(self.items):
            return None
        return self.items[self._selected_index]

class WordSimilarityApp:
    def __init__(self, master):
        self.master = master
        master.title("Do Math ON WORDS!")
        
        # Configure grid to be responsive
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(3, weight=1)

        set_appearance_mode("system")
        set_default_color_theme("dark-blue")

        self.embeddings: KeyedVectors = load_static_embeddings('embeddings/dolma_300_2024_1.2M.100_combined.txt', binary=False, no_header=True)

        self.label = CTkLabel(master, text="Enter words and operations (e.g., 'king - man + woman') and put spaces between them:", justify="center")
        self.label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")

        self.text_input = CTkEntry(master, placeholder_text="king - man + woman", justify="center")
        self.text_input.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        self.find_button = CTkButton(master, text="Find Similar Words", command=self.find_similar_words)
        self.find_button.grid(row=1, column=1, padx=(0, 20), pady=(10, 10), sticky="e")

        self.results_label = CTkLabel(master, text="Top 5 Similar Words:", justify="center")
        self.results_label.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 0), sticky="ew")

        # Create the wheel picker for displaying results
        self.wheel_picker = WheelPicker(master, items=[], width=400, height=300)
        self.wheel_picker.grid(row=3, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew")

        master.bind("<Configure>", self.update_font_size)
        self.update_font_size()

    def update_font_size(self, event=None):
        width = self.master.winfo_width()
        # Define a base font size and scale it with the window width
        base_size = max(12, int(width / 70))

        self.label.configure(font=CTkFont(size=base_size))
        self.text_input.configure(font=CTkFont(size=base_size))
        self.find_button.configure(font=CTkFont(size=base_size))
        self.results_label.configure(font=CTkFont(size=base_size))

    def find_similar_words(self):
        user_input = self.text_input.get().strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a valid input.")
            return

        try:
            similar_words = self.calculate_similar_words(user_input)
            self.display_results(similar_words)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_similar_words(self, input_text):
        text = self.parse_input(input_text)
        
        positives = []
        negatives = []
        for order, token in enumerate(text):
            if order > 0 and text[order - 1] == "-":
                negatives.append(token.lower())
            elif token not in ["+", "-", " "]:
                positives.append(token.lower())

        print(negatives)
        print(positives)
        
        similar_words = self.embeddings.most_similar(positive=positives, negative=negatives, topn=5)
        similar_words = [(word, round(score * 100, 2)) for word, score in similar_words if word not in positives + negatives]
        print(similar_words)
        
        # Store data for animations
        self.animation_data = {
            'inputs': [(word, self.embeddings[word]) for word in positives + negatives],
            'ops': self._determine_operations(text),
            'result_vector': self._calculate_result_vector(positives, negatives),
            'similar_words': similar_words
        }
        
        return similar_words
         
    def parse_input(self, input_text: str):
        # Parse the input text to extract words and operations
        # This is a placeholder for actual parsing logic
        return input_text.split()
    
    def _determine_operations(self, text):
        """Determine the sequence of operations from parsed text"""
        ops = []
        for i, token in enumerate(text):
            if token == "+":
                ops.append("add")
            elif token == "-":
                ops.append("sub")
        return ops
    
    def _calculate_result_vector(self, positives, negatives):
        """Calculate the resulting vector from positive and negative words"""
        result_vector = np.zeros(self.embeddings.vector_size)
        
        for word in positives:
            if word in self.embeddings:
                result_vector += self.embeddings[word]
        
        for word in negatives:
            if word in self.embeddings:
                result_vector -= self.embeddings[word]
                
        return result_vector
    
    def display_results(self, similar_words):
        self.wheel_picker.update_items(similar_words)
        
        # Launch animations after displaying results
        if hasattr(self, 'animation_data'):
            try:
                # Prepare data for animations
                inputs = self.animation_data['inputs']
                ops = self.animation_data['ops'] if self.animation_data['ops'] else ["add"]  # Default to add if no ops
                result_word = "Result"
                result_vector = self.animation_data['result_vector']
                similars = [(word, self.embeddings[word], score/100.0) 
                          for word, score in similar_words[:5]]
                
                # Launch animations
                run_animations(inputs, ops, (result_word, result_vector), similars)
            except Exception as e:
                print(f"Error launching animations: {e}")

if __name__ == "__main__":
    root = CTk()
    app = WordSimilarityApp(root)
    root.mainloop()