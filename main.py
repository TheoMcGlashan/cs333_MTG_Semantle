import pandas
import sys

name_weight = 1.0
type_weight = 1.0
oracle_weight = 1.0
colors_weight = 1.0
rarity_weight = 1.0
power_toughness_weight = 1.0
cmc_weight = 1.0

def main():
    if sys.argv[1] == "test":
        cards_data = pandas.read_csv("guilds-of-ravnica.csv")
    else:
        cards_data = pandas.read_csv("processed-cards.csv")

    for index, row in cards_data.iterrows():
        card_vec = card2vec(row)

        # want to write card_vec to a file alongside the card's name as an identifier.
        # need to contend with writing 30,000 vectors to a file.

def card2vec(card_data):
    name_vec = text2vec(card_data['name'])
    type_vec = type2vec(card_data['type_line'])
    oracle_vec = text2vec(card_data['oracle_text'])
    colors_vec = colors2vec(card_data['colors'])
    rarity_vec = rarity2vec(card_data['rarity'])
    power_vec = [int(card_data['power'])] if card_data['power'].isdigit() else [0]
    toughness_vec = [int(card_data['toughness'])] if card_data['toughness'].isdigit() else [0]
    cmc_vec = [int(card_data['cmc'])] if card_data['cmc'].isdigit() else [0]

    name_vec = name_weight * name_vec.normalize()
    type_vec = type_weight * type_vec.normalize()
    oracle_vec = oracle_weight * oracle_vec.normalize()
    colors_vec = colors_weight * colors_vec.normalize()
    rarity_vec = rarity_weight * rarity_vec
    power_vec = power_toughness_weight * power_vec
    toughness_vec *= power_toughness_weight
    cmc_vec *= cmc_weight
    card_vector = name_vec + type_vec + oracle_vec + colors_vec + rarity_vec + power_vec + toughness_vec + cmc_vec
    return card_vector

def text2vec(oracle_text):
    # this is the hard one.
    pass

def type2vec(type_line):
    # Card types are Creature, Instant, Sorcery, Enchantment,
    # Artifact, Planeswalker, Battle, Land.
    vec = [0, 0, 0, 0, 0, 0, 0, 0]
    if "Creature" in type_line:
        vec[0] = 1
    if "Instant" in type_line:
        vec[1] = 1
    if "Sorcery" in type_line:
        vec[2] = 1
    if "Enchantment" in type_line:
        vec[3] = 1
    if "Artifact" in type_line:
        vec[4] = 1
    if "Planeswalker" in type_line:
        vec[5] = 1
    if "Battle" in type_line:
        vec[6] = 1
    if "Land" in type_line:
        vec[7] = 1

    return vec

def colors2vec(colors):
    # Colors are W, U, B, R, G
    vec = [0, 0, 0, 0, 0]
    if "W" in colors:
        vec[0] = 1
    if "U" in colors:
        vec[1] = 1
    if "B" in colors:
        vec[2] = 1
    if "R" in colors:
        vec[3] = 1
    if "G" in colors:
        vec[4] = 1

    return vec

def rarity2vec(rarity):
    if rarity == 'common':
        vec = [1]
    elif rarity == 'uncommon':
        vec = [2]
    elif rarity == 'rare':
        vec = [3]
    elif rarity == 'mythic':
        vec = [4]
    else:
        vec = [0]
    return vec

if __name__ == "__main__":
    main()
