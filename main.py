import argparse
import random
from datetime import date
from tkinter import *
import sys
import os

import pandas
import pyautogui

from semantle import semantle

# stuff to make it hopefully compile to one file
if getattr(sys, "frozen", False):
    os.chdir(sys._MEIPASS)

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

    print(cardname)

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
    root.configure(bg="#c56626")

    # create a blue text box with welcome message
    welcome_message = "Welcome to MTG Semantle! Each day, a random commander-legal Magic: the Gathering card is chosen, and you must guess the card. For incorrect guesses, the similarity between your guess and the correct card will be shown."
    height = int(int(height) / 140)
    text = Text(root, wrap=WORD, height=height)
    text.pack(padx=10, pady=10, expand=False, fill=BOTH)
    text.configure(bg="#ADD8E6")
    text.insert(index="1.0", chars=welcome_message)
    text["state"] = "disabled"

    # create a frame so that the button appears above the results
    controls_frame = Frame(root, bg=root["bg"])
    controls_frame.pack(fill=X, padx=10, pady=8)

    # create an entry box for card name input
    entry = Entry(controls_frame, width=20)
    entry.pack(padx=(0, 6))

    # create a frame to hold the results
    result_frame = Frame(root, bg=root["bg"])
    result_frame.pack(fill=BOTH, expand=True, padx=10, pady=8)

    guesses = {}

    def clicked():
        # get the name from the entry box
        name = entry.get().strip()
        # if the name is empty, do nothing
        if not name:
            return

        # process the guess and get the rank
        rank = process_guess(name, df)
        # determine the color based on the rank
        if rank is None:
            color = "#FFB6B6"
        else:
            try:
                total = len(df)
                ratio = 0.0 if total <= 1 else (rank - 1) / (total - 1)
            except Exception:
                ratio = 1.0
            g_r, g_g, g_b = (0x90, 0xEE, 0x90)
            r_r, r_g, r_b = (0xFF, 0xB6, 0xB6)
            ir = int(g_r * (1 - ratio) + r_r * ratio)
            ig = int(g_g * (1 - ratio) + r_g * ratio)
            ib = int(g_b * (1 - ratio) + r_b * ratio)
            color = "#{:02X}{:02X}{:02X}".format(ir, ig, ib)

        # store the guess so we can sort after each guess
        guesses[name] = (rank, color)

        # clear previous results
        for child in result_frame.winfo_children():
            child.destroy()

        # sort guesses by rank
        sorted_items = sorted(
            guesses.items(),
            key=lambda kv: (kv[1][0] if kv[1][0] is not None else float("inf")),
        )

        # display each guess with its rank and color
        for g_name, (g_rank, g_color) in sorted_items:
            if g_rank is None:
                text_val = f"'{g_name}': Card not found."
            elif g_rank == 1:
                text_val = f"'{g_name}': Correct! You found the card!"
            else:
                text_val = f"'{g_name}': ranked {g_rank} out of {len(df)} cards."
            lbl = Label(
                result_frame,
                text=text_val,
                bg=g_color,
                fg="black",
                anchor="w",
                justify=LEFT,
            )
            lbl.pack(fill=X, padx=5, pady=2)

    # create the button that processes guesses
    button = Button(controls_frame, text="Enter a card name", bg="#ADD8E6", command=clicked)
    button.pack()

    # Execute Tkinter
    root.mainloop()


def process_guess(cardname, df):
    # get the rank of the guessed cardname
    try:
        matches = [idx for idx in df.index if cardname.lower() in idx.lower()]
    except Exception:
        return None
    if not matches:
        return None
    return int(df.index.get_loc(matches[0])) + 1

# get half of the screen size, for the gui window
def get_geometry():
    screen_size = pyautogui.size()
    width = str(int(screen_size[0] / 2))
    height = str(int(screen_size[1] / 2))
    geometry = width + "x" + height

    return geometry, height


if __name__ == "__main__":
    main()