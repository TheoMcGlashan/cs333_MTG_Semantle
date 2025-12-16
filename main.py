import random
from datetime import date
from tkinter import *

import pandas
from pyautogui import size


def main():
    # get half the screen size
    geometry = get_geometry()

    # get a random mtg cardname based on today's date
    cardname = get_cardname()

    # create the gui window with the correct size
    create_gui(geometry)


def get_cardname():
    # create a random seed based on today's date
    date_string = str(date.today())
    day, month, year = date_string.split("-")
    seed = int(str(int(year) - 2000) + str(month) + str(day))

    # read in dataset of names
    names = pandas.read_csv("names.csv")

    # chose a cardname based on the random seed
    random.seed(seed)
    rand_index = random.randint(0, len(names) - 1)
    cardname = names.iloc[rand_index]

    return cardname["name"]


def create_gui(geometry):
    # create root window
    root = Tk()

    # root window title and dimension
    root.title("Magic: the Gathering Semantle")
    # Set geometry to half of screen size
    root.geometry(geometry)

    # Execute Tkinter
    root.mainloop()


# get half of the screen size, for the gui window
def get_geometry():
    screen_size = size()
    width = str(int(screen_size[0] / 2))
    height = str(int(screen_size[1] / 2))
    geometry = width + "x" + height

    return geometry


if __name__ == "__main__":
    main()
