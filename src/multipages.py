import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Button
from tkinter import *
from submain import *
import os

titleFont =("Times", 50)


class tkinterApp(tk.Tk):

    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Face Recognition using Eigenface")
        self.geometry("1200x720")
        # creating a container
        container = tk.Frame(self)
        container.pack(side= "top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight = 1)

        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for i in (StartPage, MainPage, BonusPage):
            frame = i(container,self)

            # initializing frame of that object from
            # startpage, mainpage, bonuspage respectively with
            # for loop
            self.frames[i] = frame

            frame.grid(row = 0, column = 0, sticky = "nsew")

        # to display the current frame passed as
        # parameter
        self.show_frame(StartPage)

    def show_frame(self,cont):
        frame = self.frames[cont]
        frame.tkraise()

# First Window Start Page
class StartPage(tk.Frame):
    def __init__ (self, parent, controller):
        tk.Frame.__init__(self,parent)

        startPageBg = "#ececec"


        dirname = os.path.dirname(__file__)
        welcomeImagePath = os.path.join(dirname, "assets\image.jpg")

        hahaimage = Image.open(welcomeImagePath)
        displayImage = ImageTk.PhotoImage(hahaimage)

        label = Label(image = displayImage)
        label.pack()




        label = ttk.Label(self, text="Start Page", font = titleFont)

        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Main Page", command = lambda : controller.show_frame(MainPage))

        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        button2 = ttk.Button(self, text ="Bonus Page",command = lambda : controller.show_frame(BonusPage))

        button2.grid(row = 2, column = 1, padx = 10, pady = 10)

        # dirname = os.path.dirname(__file__)
        # welcomeImagePath = os.path.join(dirname, "assets\profile_icon.png")

        # hahaimage = Image.open(welcomeImagePath)
        # displayImage = ImageTk.PhotoImage(hahaimage)

        # startImage = ttk.Label(self, image = displayImage)
        # startImage.grid(row = 3, column = 1)

# Second Window Start Page
class MainPage(tk.Frame):
    def __init__ (self, parent, controller):

        dirname = os.path.dirname(__file__)
        welcomeImagePath = os.path.join(dirname, "assets\profile_icon.png")

        hahaimage = Image.open(welcomeImagePath)
        displayImage = ImageTk.PhotoImage(hahaimage)

        label = Label(image = displayImage)

        label.pack()
        tk.Frame.__init__(self,parent)

        label = ttk.Label(self, text="Main Page", font = titleFont)

        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Start Page", command = lambda : controller.show_frame(StartPage))

        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
        button2 = ttk.Button(self, text ="Bonus Page",command = lambda : controller.show_frame(BonusPage))

        button2.grid(row = 2, column = 1, padx = 10, pady = 10)

# Third Window Start Page
class BonusPage(tk.Frame):
    def __init__ (self, parent, controller):
        tk.Frame.__init__(self,parent)

        label = ttk.Label(self, text="Bonus Page", font = titleFont)

        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Start Page", command = lambda : controller.show_frame(StartPage))

        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
        button2 = ttk.Button(self, text ="Main Page",command = lambda : controller.show_frame(MainPage))

        button2.grid(row = 2, column = 1, padx = 10, pady = 10)      

app = tkinterApp()
app.mainloop()
