import pandas
import sys
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

type_weight = 1.0
text_weight = 1.0
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
        quit()

        # want to write card_vec to a file alongside the card's name as an identifier.
        # need to contend with writing 30,000 vectors to a file.

def card2vec(card_data):
    type_vec = type2vec(card_data['type_line'])
    text_vec = text2vec(card_data['name'] + ": " + card_data['oracle_text'])
    colors_vec = colors2vec(card_data['colors'])
    rarity_vec = rarity2vec(card_data['rarity'])
    if card_data['power'].isdigit():
        power_vec = card_data['power']
    else:
        power_vec = [0]
    if card_data['cmc'].isdigit():
        cmc_vec = card_data['cmc']
    else:
        cmc_vec = [0]
    if card_data['toughness'].isdigit():
        toughness_vec = card_data['toughness']
    else:
        toughness_vec = [0]

    
    type_vec = type_weight * type_vec.normalize()
    text_vec = text_weight * text_vec.normalize()
    colors_vec = colors_weight * colors_vec.normalize()
    rarity_vec = rarity_weight * rarity_vec
    power_vec *= power_toughness_weight
    toughness_vec *= power_toughness_weight
    cmc_vec *= cmc_weight
    card_vector = type_vec + text_vec + colors_vec + rarity_vec + power_vec + toughness_vec + cmc_vec
    return card_vector

def text2vec(text):

    # preproces the documents, and create TaggedDocuments
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                tags=[str(i)]) for i,
                doc in enumerate(text)]

    # train the Doc2vec model
    model = Doc2Vec(vector_size=20,
                    min_count=2, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    # get the document vectors
    document_vectors = [model.infer_vector(
        word_tokenize(doc.lower())) for doc in text]

    #  print the document vectors
    for i, doc in enumerate(text):
        print("Document", i+1, ":", doc)
        print("Vector:", document_vectors[i])
        print()
    return document_vectors[0]

def type2vec(type_line):
    # Card types are Creature, Instant, Sorcery, Enchantment,
    # Artifact, Planeswalker, Battle, Land, kindred, basic, snow, legendary
    vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
    if "Basic" in type_line:
        vec[8] = 1
    if "Snow" in type_line:
        vec[9] = 1
    if "Legendary" in type_line:
        vec[10] = 1
    if "Kindred" in type_line:
        vec[11] = 1
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
