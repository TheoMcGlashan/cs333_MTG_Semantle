import pandas
import sys
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt_tab')
from sklearn.preprocessing import normalize
import numpy as np
from math import log, isnan
import argparse

#Default weights for embedding catagories
TYPE_WEIGHT = 1
TEXT_WEIGHT = 1.0
MANA_WEIGHT = 1.0
RARITY_WEIGHT = 1.0
POWER_WEIGHT = 1.0
TOUGHNESS_WEIGHT= 1.0
CMC_WEIGHT = 1.0

#Creates a parser for command line arguments
parser = argparse.ArgumentParser(prog='card2vec',
                    description='Converts magic cards to vectors and adjusts based on weights',
                    epilog='<3')
parser.add_argument("filename")
parser.add_argument('-v', '--vector', action="store_true", help = "convert cards to vector")
parser.add_argument('-n', '--norm', action="store_true", help = "run weights through vector")
parser.add_argument('-y','--type', type=float, default= TYPE_WEIGHT, help= "weight for type embeddings")
parser.add_argument('-x','--text', type=float, default= TEXT_WEIGHT, help= "weight for text embeddings")
parser.add_argument('-m','--mana', type=float, default= MANA_WEIGHT, help= "weight for mana embeddings")
parser.add_argument('-r','--rarity', type=float, default= RARITY_WEIGHT, help= "weight for rarity embedding")
parser.add_argument('-p','--power', type=float, default= POWER_WEIGHT, help= "weight for power embedding")
parser.add_argument('-t','--toughness', type=float, default= POWER_WEIGHT, help= "weight for toughness embedding")
parser.add_argument('-c','--cmc', type=float, default= CMC_WEIGHT, help= "weight for cmc embedding")


def main():
    """
    Takes in a file name for cards and converts them into vectors/applies weights

    Usage: 'filename' 'mode' 'options'

    Modes:
        -v ->
            Inputs: CSV file 'filename'.csv with one line per card with the following header: 
            name,mana_cost,type_line,oracle_text,colors,set_name,rarity,artist,power,toughness,cmc,keywords

            Outputs: CSV file: vectorized-'filename' with one line per card in vector representation with the following header:
            Creature,Instant,Sorcery,Enchantment,Artifact,Planeswalker,Battle,Land,Kindred,Basic,Snow,Legendary,text_1,text_2,text_3,text_4,text_5,text_6,text_7,text_8,text_9,text_10,text_11,text_12,text_13,text_14,text_15,text_16,text_17,text_18,text_19,text_20,W,U,B,R,G,C,Generic,X,Phyrexian,Snow_Mana,Pips,Rarity,Power,Toughness,CMC,Name

        -n -> 
            Inputs: CSV file: vectorized-'filename' from 'v' mode with respective header.

            Optional inputs: --type, --text, --mana, --rarity, --pt, --cmc

            Outputs: CSV file: weighted-'filename' with embeddings transformed via programmed weights

    Options:

        Can be used to manually set weights in -n mode

        -y --type weight for type embeddings

        -x --text weight for text embeddings

        -m --mana weight for mana embeddings

        -r --rarity weight for rarity embedding

        -p --power weight for power embedding

        -t --toughness weight for toughness embedding

        -c --cmc weight for cmc embedding


    """
    #Names for each embedding, assumed same on input for -v
    colnames = ['Creature', 'Instant', 'Sorcery', 'Enchantment','Artifact', 'Planeswalker', 'Battle', 'Land', 'Kindred', 'Basic', 'Snow', 'Legendary',
                    'text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10','text_11','text_12','text_13','text_14','text_15','text_16','text_17','text_18','text_19','text_20',
                    'W','U','B','R','G','C','Generic','X','Phyrexian','Snow_Mana','Pips','Rarity','Power','Toughness','CMC','Name']
    

    #save file prefix for post vector cards and post weighted cards
    v_prefix = "vector_"
    n_prefix = "weighted_"

    #parses args, see above for what values go to what arg name
    args = parser.parse_args()
    #print(parser.parse_args())
    file = args.filename


    
    if args.vector:
        cards_data = pandas.read_csv(file)
        
        card_vecs = []
        for index, row in cards_data.iterrows():    #for each card, convert it to a vector and append to card_vecs
            card_vec = card2vec(row)
            card_vecs.append(card_vec)
        
        save_name = v_prefix+file
        pandas.DataFrame(card_vecs).set_axis(colnames, axis=1).to_csv(save_name, index=0)

    if args.norm:
        cards = pandas.read_csv(v_prefix+file)
        if list(cards.columns) != colnames:           #checks that header matches expected
            print("Bad header")
            print(list(cards.columns))
        else:
            cards = weight_cards(cards, args)          #weight_cards does all weights and normalizations
            save_name = n_prefix+file[len(v_prefix):len(file)]
            cards.to_csv(save_name, index=0)


def weight_cards(card_data, args):


    #Adjusting all type embeddings by type_weight
    card_data.loc[:, :'Legendary']*=args.type
    #print(card_data.loc[:, :'Legendary'])

    #Adjusting all text by text_weight
    card_data.loc[:, 'text_1':'text_20']*=args.text
    #print(card_data.loc[:, 'text_1':'text_20'])

    #Adjusting all mana value embeddings by colors_weight
    card_data.loc[:, 'W':'Pips']*=args.mana
    #print(card_data.loc[:, 'W':'Pips'])

    #Adjusting rarity embeddings text by rarity_weight
    card_data.loc[:,'Rarity']*=args.rarity
    #print(card_data.loc[:,'Rarity'])

    #Adjusting all power embeddings by power_weight
    card_data.loc[:, 'Power']*=args.power
    #print(card_data.loc[:, 'text_1':'text_20'])

    #Adjusting all toughness embeddings by power_weight
    card_data.loc[:, 'Toughness']*=args.toughness
    #print(card_data.loc[:, 'text_1':'text_20'])

    #Adjusting all cmc embeddings by cmc_weight
    card_data.loc[:, 'CMC']*=args.cmc
    #print(card_data.loc[:, 'CMC'])


    return card_data

def card2vec(card_data):
    """Takes a cards data and converts it into 7 different vectors and then appends them:

    Types line, text box (including type line and name), mana value, rarity, power, toughness, cmc

    adds name at the end
    """
    type_vec = type2vec(card_data['type_line'])
    text_vec = text2vec((card_data['name']) + ": " +  (card_data['type_line']) + " : " + str(card_data['oracle_text']))
    mana_vec = mana2vec(card_data['mana_cost'])
    rarity_vec = rarity2vec(card_data['rarity'])
    if card_data['power'].is_integer():
        power_vec = [card_data['power']]
    else:
        power_vec = [0]
    if card_data['cmc'].is_integer():
        cmc_vec = [card_data['cmc']]
    else:
        cmc_vec = [0]
    if card_data['toughness'].is_integer():
        toughness_vec = [card_data['toughness']]
    else:
        toughness_vec = [0]

    card_vec = type_vec + text_vec.tolist() + mana_vec + rarity_vec + power_vec + toughness_vec + cmc_vec  

    card_vec.append(card_data['name'])
    return card_vec

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

    return document_vectors[0]

def type2vec(type_line):
    """
    Maps type to binary emebeddings in the following order:

    Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Battle, Land, kindred, basic, snow, legendary
    """
    # Card types are Creature, Instant, Sorcery, Enchantment,
    # Artifact, Planeswalker, Battle, Land, kindred, basic, snow, legendary
    vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

def mana2vec(mana_cost):
    """
    Converts mana cost to vector

    Embeddings are: W, U, B, R, G, C, Gen, X, Phy, S, Pips
    
    For specific symbols, the corrosponding embedding is increased by 1. Ex: {W} -> [+1,+0,...], with the same for X symbols, pyrexian etc
     
    Pips is the total number of {} in the mana cost, so generic, X, etc are counted as pips.
     
    Generic is the total generic that could be used to cast the spell, Reaper King has Generic = 10 
    """
    vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    try:
        mana_cost = mana_cost.split("{")
    except AttributeError:
        return vec
    for i in range(len(mana_cost)):
        mana_cost[i] = mana_cost[i].strip("}")
    # embeddings are W, U, B, R, G, C, Gen, X, Phy, S, Pips
    vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
    for i in mana_cost:
        vec[10] += 1
        if "W" == i:
            vec[0] += 1
        elif "U" == i:
            vec[1] += 1
        elif "B" == i:
            vec[2] += 1
        elif "R" == i:
            vec[3] += 1
        elif "G" == i:
            vec[4] += 1
        elif "C" == i:
            vec[5] += 1
        elif i.isdigit():
            vec[6] = float(i)
        elif "X" == i:
            vec[7] += 1
        elif "P" in i:
            vec[8] += 1
            if "W" in i:
                vec[0] += 1
            if "U" == i:
                vec[1] += 1
            if "B" == i:
                vec[2] += 1
            if "R" == i:
                vec[3] += 1
            if "G" == i:
             vec[4] += 1
        elif "S" in i:
            vec[9] += 1
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

