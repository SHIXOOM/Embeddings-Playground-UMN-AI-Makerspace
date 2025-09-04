from tkinter import messagebox
from customtkinter import CTk, CTkLabel, CTkEntry, CTkButton, CTkTextbox, set_appearance_mode, set_default_color_theme, CTkFont
from gensim.models.keyedvectors import KeyedVectors
from embeddings_loader import load_static_embeddings

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

        self.results_display = CTkTextbox(master)
        self.results_display.grid(row=3, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew")
        self.results_display.configure(state='disabled')
        self.results_display.tag_config("center", justify="center")

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
        self.results_display.configure(font=CTkFont(size=base_size))

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
        return similar_words
         
    def parse_input(self, input_text: str):
        # Parse the input text to extract words and operations
        # This is a placeholder for actual parsing logic
        return input_text.split()
    
    def display_results(self, similar_words):
        self.results_display.configure(state='normal')
        self.results_display.delete("1.0", "end")
        for word, score in similar_words:
            self.results_display.insert("end", f"{word}: {score}\n", "center")
        self.results_display.configure(state='disabled')

if __name__ == "__main__":
    root = CTk()
    app = WordSimilarityApp(root)
    root.mainloop()