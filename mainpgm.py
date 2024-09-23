# Classification-of-species-of-Iris-dataset-
Classification of species of Iris dataset  in machine learning and integrating with the tkinter
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

class SpeciesClassifierApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Species Classifier")
        
        # Initialize model
        self.model = self.create_model()
        
        # Create GUI components
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)
        
        self.img_label = tk.Label(root)
        self.img_label = tk.Label(root)
        self.img_label.pack()
        
        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation='softmax')  # Assuming 3 classes for species
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Load and preprocess the image
        img = Image.open(file_path)
        img = img.resize((150, 150), Image.LANCZOS)
        img_array = np.array(img) / 255.0  # Normalize the image

        if img_array.shape != (150, 150, 3):
            messagebox.showerror("Error", "Image must be in RGB format with size 150x150.")
            return

        img_array = np.expand_dims(img_array, axis=0)

        # Display the image in the GUI
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk
        
        # Make predictions using the model
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Map predicted class to species names
        species = ['setosa', 'versicolor', 'virginica']
        predicted_species = species[predicted_class[0]]

        # Display the result in the interface
        self.result_label.config(text=f'The predicted species is: {predicted_species}')

if _name_ == "_main_":
    root = tk.Tk()
    app = SpeciesClassifierApp(root)
    root.mainloop()
