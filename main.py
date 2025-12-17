import argparse
import random
from datetime import date
from tkinter import *

import pandas
import pyautogui

from semantle import semantle

parser = argparse.ArgumentParser(prog="1", description="2", epilog="<3")
parser.add_argument("-f", "--file", type=str, default="processed-cards.csv")
parser.add_argument("-c", "--card", type=str, default="Colossal Dreadmaw")
parser.add_argument("-d", "--debug", type=bool, default=False, help="debug")


def main():
    args = parser.parse_args()
    # get half the screen size
    geometry, height = get_geometry()

    # get a random mtg cardname based on today's date
    cardname = get_cardname()

    df = semantle(args.file, cardname, savefile=True)

    # create the gui window with the correct size
    create_gui(geometry, height, df)


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


def create_gui(geometry, height, df):
    # create root window
    root = Tk()

    # create root node with green background
    root.title("Magic: the Gathering Semantle")
    root.geometry(geometry)
    root.configure(bg="#90ee90")

    # create a blue text box with welcome mesage
    welcome_message = "Welcome to MTG Semantle! Each day, a random commander-legal Magic: the Gathering card is chosen, and you must guess the card. For incorrect guesses, the similarity between your guess and the correct card will be shown."
    height = int(int(height) / 140)
    text = Text(root, wrap=WORD, height=height)
    text.pack(padx=10, pady=10, expand=False, fill=BOTH)
    text.configure(bg="#ADD8E6")
    text.insert(index="1.0", chars=welcome_message)
    text["state"] = "disabled"

    # create an entry box for users to input cardname
    label = Label(root, text="Enter a card name")
    label.pack()
    label.configure(bg="#90ee90")
    entry = Entry(root)
    entry.pack()

    def clicked():
        rank = process_guess(entry.get(), df)

    button = Button(root, text="Enter a card name", bg="#ADD8E6", command=clicked)
    button.pack()

    def card_entered():
        pass

    # Execute Tkinter
    root.mainloop()


def process_guess(cardname, df):
    print(df.columns)
    row = df[df["Card_name"] == cardname]
    print(row)

    return row


# get half of the screen size, for the gui window
def get_geometry():
    screen_size = pyautogui.size()
    width = str(int(screen_size[0] / 2))
    height = str(int(screen_size[1] / 2))
    geometry = width + "x" + height

    return geometry, height


if __name__ == "__main__":
    main()
