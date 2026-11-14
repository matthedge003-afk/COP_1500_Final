import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
from Network import Network
from Vector import Vector

class MNISTCanvas:
    def __init__(self,root, network: 'Network'):
        self.network = network

        self.root = root
        self.root.title("DRAWING BOARD")

        self.canvas_size = 560
        self.mnist_size = (14, 14)

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=10)

        self.image = Image.new("L",(self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.process_drawing)

        self.clear_btn = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(pady=5)

        print("Draw a digit (0-9) on the blackboard!")

    def paint(self, event):
        r = 10
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)

        self.canvas.create_oval(x1,y1, x2,y2, fill="white", outline="white")
        self.draw.ellipse([x1,y1, x2,y2], fill="white", outline="white")

    def process_drawing(self, event=None):
        # 1. Apply FUZZ
        # fuzzy_image = self.image.filter(ImageFilter.GaussianBlur(radius=2))

        # 2. Resize
        img_resized = self.image.resize(self.mnist_size, Image.Resampling.BILINEAR)

        # flatten
        raw_pixels = list(img_resized.getdata())
        mnist_flat_list = [pixel / 255.0 for pixel in raw_pixels]

        # NETWORK TIME
        output = self.network.feedforward(Vector(mnist_flat_list))

        highest_index = 0
        for i in range(output.length):
            if output[i] > output[highest_index]:
                highest_index = i


        int_to_word = {
            0: "Zero",
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six", 
            7: "Seven",
            8: "Eight",
            9: "Nine"
        }

        print("Guess:", int_to_word[highest_index])

    
        print("Output:", output)


    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    network = Network.load("blurry_mnist_network1.pkl")

    root = tk.Tk()
    app = MNISTCanvas(root, network)
    root.mainloop()
