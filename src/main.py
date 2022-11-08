import tkinter as tk
from tkinter import font as tkfont
from tkinter import *
from PIL import ImageTk, Image
import os


window = tk.Tk()

dirname = os.path.dirname(__file__)
imagePath1 = os.path.join(dirname, 'assets\profile_icon.png')
imagePath2 = os.path.join(dirname, 'assets\profile_icon.png')


# App Title and Size
window.title('Face Recognition using Eigenface')
window.geometry("1200x720")

# Displaying Title
appTitle = Label (window, text = "Face Recognition")
appTitle.config(font= ("Courier", 30))
appTitle.pack(pady = 30)


# Create leftContainer
leftContainer = Frame(window)




leftContainer.pack(side=tk.LEFT)


# Create rightContainer

rightContainer = Frame(window)

# Create an object of tkinter ImageTk


# Create a Label Widget to display image

imageContainer = Frame(rightContainer)

img1 =Image.open(imagePath1)
img2 = Image.open(imagePath2)

displayImg1 = ImageTk.PhotoImage(img1.resize((275, 275), Image.ANTIALIAS))
displayImg2 = ImageTk.PhotoImage(img1.resize((275, 275), Image.ANTIALIAS))

# Contain Image Border and Image 1
imageBorder1 = Frame(imageContainer, background= "black")

labelImg1 = Label(imageBorder1, image = displayImg1, bd = 5, background="#e0e0e0")
labelImg1.place(relx=0.05, rely=0.5,)   
labelImg1.pack(padx = 2, pady = 2)

imageBorder1.pack(padx = 20, pady = 20, side=tk.LEFT)

# Contain Image Border and Image 2
imageBorder2 = Frame(imageContainer, background= "black")

labelImg2 = Label(imageBorder2, image = displayImg2, bd = 5, background="#e0e0e0")
labelImg2.place(relx=0.5, rely=0.5)
labelImg2.pack(padx = 2, pady = 2)

imageBorder2.pack(padx = 20, pady = 20, side=tk.RIGHT)

imageContainer.pack(side = tk.BOTTOM)
 
rightContainer.pack(padx = 150, pady = 100, side=tk.RIGHT)

window.mainloop()