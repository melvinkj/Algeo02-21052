import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Button
from tkinter import *
from submain import *
# from cam import *
import os


def WelcomePage():

    window = Tk()
    window.title('Face Recognition using Eigenface - Welcome Page')
    window.geometry("1200x720")
    bgColor = "#ececec"
    window.configure(background= bgColor)


    # leftContainer
    leftContainer = Frame(window)
    dirname = os.path.dirname(__file__)
    welcomeImagePath = os.path.join(dirname, 'assets\image.jpg')


    displayImage = ImageTk.PhotoImage(Image.open(welcomeImagePath))

    Label(leftContainer, image = displayImage).pack(padx=(30,0))

    leftContainer.pack(side= tk.LEFT)

    # rightContainer
    rightContainer = Frame(window, background= bgColor)
    rightContainer.grid_rowconfigure(3, weight=1)
    rightContainer.grid_columnconfigure(0, weight=1)

    title = Label(rightContainer, text = "MKJ", font=("Times", 50), background= bgColor)
    title.grid(row = 0, column = 0, pady = (20, 20))

    subtitle = Label(rightContainer, text = "Face Recognition", font=("Times", 20), background= bgColor)
    subtitle.grid(row = 1, column = 0, pady= (0, 50))

    mainPageBtn = Button(rightContainer, text = "Go To Main Page", font=("Times", 20), background= bgColor, command = window.destroy)
    mainPageBtn.grid(row = 2, column= 0)
    
    rightContainer.pack(pady=(100, 0))



    window.mainloop()

def MainPage():
    # Set default image
    dirname = os.path.dirname(__file__)
    imagePath1 = os.path.join(dirname, 'assets\profile_icon.png')
    imagePath2 = os.path.join(dirname, 'assets\profile_icon.png')

    executionTime = "00:00:00" #ini nanti diubah ke yang asli
    result = "None"
    dataSetName = "No Folder Selected"
    userImageName = "No File Selected"

    window = Tk()

    # App Title and Size
    bgColor = "#fffde7"
    window.title('Face Recognition using Eigenface - Main Page')
    window.geometry("1200x720")
    window.configure(background= bgColor)

    # Displaying Title
    appTitle = Label (window, text = "Face Recognition", font=('Times', 30), background="#f6e3ba", height=3)
    appTitle.pack(fill = 'x')           # Fill = 'x', to set width 100% screen


    # Create leftContainer
    leftContainer = Frame(window, background= bgColor)
    leftContainer.grid_rowconfigure(3, weight=1)
    leftContainer.grid_columnconfigure(0, weight=1)

    # Insert Dataset container
    dataSetContainer = Frame(leftContainer, background= bgColor)

    Label(dataSetContainer, text= "Insert Your Dataset", font=("Times",18), background= bgColor).pack()

    dataSetBtn = Button(dataSetContainer, text = 'Choose Folder',height=2, width= 15, background='#fffde7')
    dataSetBtn.bind("<Enter>", on_enter)
    dataSetBtn.bind("<Leave>", on_leave)
    dataSetBtn.pack(side = "left", pady=10)

    dataSetContainer.grid(row = 0, column = 0, pady = 2, padx = 20, sticky= "e")

    #Insert Image container
    userImageContainer = Frame(leftContainer, background= bgColor)

    Label(userImageContainer, text= "Insert Your Image", font=("Times",18), background= bgColor).pack()

    userImageBtn = Button(userImageContainer, text = 'Choose Image',height=2, width= 15, background='#fffde7')
    userImageBtn.bind("<Enter>", on_enter)
    userImageBtn.bind("<Leave>", on_leave)
    userImageBtn.pack(side = "left", pady=10)

    userImageContainer.grid(row = 1, column = 0, pady = 2)

    # Start Button
    startBtn = Button(leftContainer, text = "Start Program", font=("Times", 22), background="#decdb9", width= 15)

    startBtn.bind("<Enter>", on_start_enter)
    startBtn.bind("<Leave>", on_start_leave)

    startBtn.grid(row = 2, column = 0, pady = 10)

    # Result Container
    resultContainer = Frame(leftContainer, background= bgColor)
    Label (resultContainer, text = "Result : ", font=('Times', 23), background= bgColor).pack(pady= (15,0))
    Label (resultContainer, text = result, font=('Times', 16), background= bgColor).pack()

    resultContainer.grid(row = 3, column = 0, pady = 2)

    leftContainer.pack(padx = 60, side=tk.LEFT)


    # Create rightContainer
    rightContainer = Frame(window, background= bgColor)

    # Container for Displaying Image

    displayImagesContainer = Frame(rightContainer, background= bgColor)

    # Create an object of tkinter ImageTk
    displayImg1 = readImage(imagePath1)
    displayImg2 = readImage(imagePath2)

    # Container for Test Image
    testImageContainer = Frame(displayImagesContainer, background= bgColor)

    Label(testImageContainer, text = "Test Image" , font=('Times', 18), background= bgColor).pack(side = tk.TOP, pady=5)

    # Contain Image Border and Image 1
    imageBorder1 = Frame(testImageContainer, background= "black")

    labelImg1 = Label(imageBorder1, image = displayImg1, bd = 5, background="#ffefcb")
    labelImg1.place(relx=0.05, rely=0.5,)   
    labelImg1.pack(padx = 2, pady = 2)

    imageBorder1.pack(padx = 20, side=tk.BOTTOM)
    testImageContainer.pack(side = tk.LEFT)

    # Container for Closest Image
    closestImageContainer = Frame(displayImagesContainer, background= bgColor)

    Label(closestImageContainer, text = "Closest Result", font=('Times', 18), background= bgColor).pack(side = tk.TOP, pady=5)

    # Contain Image Border and Image 2
    imageBorder2 = Frame(closestImageContainer, background= "black")

    labelImg2 = Label(imageBorder2, image = displayImg2, bd = 5, background="#ffefcb")
    labelImg2.place(relx=0.5, rely=0.5)
    labelImg2.pack(padx = 2, pady = 2)

    imageBorder2.pack(padx = 20, side=tk.BOTTOM)
    closestImageContainer.pack(side = tk.RIGHT)

    displayImagesContainer.pack(side = tk.TOP)

    # Execution time

    timeContainer = Frame(rightContainer, background= bgColor)

    Label (timeContainer, text = "Execution Time : ", font=('Times', 18), background= bgColor).pack(pady= (30,0))
    Label (timeContainer, text = executionTime, font=('Times', 20), background= bgColor).pack()

    timeContainer.pack(side = tk.BOTTOM)
    
    rightContainer.pack(padx = 40, side=tk.RIGHT)

    window.mainloop()

# Main

WelcomePage()
MainPage()
show_frame()