# Berisi fungsi dan prosedur yang digunakan untuk menunjang main

from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image

# FUNGSI 
# Membaca image pada path tertentu
# Mengembalikan display image yang telah diresize dan dapat digunakan oleh tkinter
def readImage(path):
    image = Image.open(path)
    displayimage = ImageTk.PhotoImage(image.resize((275, 275), Image.ANTIALIAS))
    return displayimage

# PROSEDUR
# Prosedur prosedur dibawah ini digunakan untuk hover
# on_enter, saat cursor mouse diatas button
# on_leave, saat cursor mouse diluar button

def on_enter(e):
    e.widget['background'] = '#dfcaa0'

def on_leave(e):
    e.widget['background'] = '#fffde7'

def on_start_enter(e):
    e.widget['background'] = '#b58e62'

def on_start_leave(e):
    e.widget['background'] = "#decdb9"