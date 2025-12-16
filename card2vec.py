import pandas
import sys
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt_tab')
from sklearn.preprocessing import normalize
import numpy as np
from math import log, isnan, sqrt
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
    colnames = ['Creature', 'Instant', 'Sorcery', 'Enchantment','Artifact', 'Planeswalker', 'Land', 'Legendary',
                    'text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10','text_11','text_12','text_13','text_14','text_15','text_16','text_17','text_18','text_19','text_20','text_21','text_22','text_23','text_24','text_25','text_26','text_27','text_28','text_29','text_30',
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
        
        card_vecs = card2vec(cards_data)
        
        save_name = v_prefix+file
        pandas.DataFrame(card_vecs).set_axis(colnames, axis=1).to_csv(save_name, index=0)

    if args.norm:
        cards = pandas.read_csv(v_prefix+file)
        if list(cards.columns) != colnames:           #checks that header matches expected
            print("Bad header")
            print(list(cards.columns))
        else:
            cards = weight_cards(cards, args)          #weight_cards does all weights and normalizations
            save_name = n_prefix+file
            cards.to_csv(save_name, index=0)


def weight_cards(card_data, args):


    #Adjusting all type embeddings by type_weight
    card_data.loc[:, :'Legendary']*=args.type
    card_data.loc[:, :'Legendary']= weight_section(card_data.loc[:, :'Legendary'],8)
    card_data.loc[:, :'Legendary']*=args.type

    #Adjusting all text by text_weight
    card_data.loc[:, 'text_1':'text_30']*=args.text

    #Adjusting all mana value embeddings by colors_weight
    card_data.loc[:, 'W':'Pips']*=args.mana
    card_data.loc[:, 'W':'Pips'] = weight_section(card_data.loc[:, 'W':'Pips'],11)
    card_data.loc[:, 'W':'Pips']*=args.mana
    

    #Adjusting rarity embeddings text by rarity_weight
    card_data.loc[:,'Rarity']*=args.rarity
    card_data.loc[:,'Rarity'] = weight_section(card_data.loc[:,'Rarity'],1)
    card_data.loc[:,'Rarity']*=args.rarity

    #Adjusting all power embeddings by power_weight
    card_data.loc[:, 'Power']*=args.power
    card_data.loc[:, 'Power'] = weight_section(card_data.loc[:, 'Power'],1)
    card_data.loc[:, 'Power']*=args.power

    #Adjusting all toughness embeddings by power_weight
    card_data.loc[:, 'Toughness']*=args.toughness
    card_data.loc[:, 'Toughness'] = weight_section(card_data.loc[:, 'Toughness'],1)
    card_data.loc[:, 'Toughness']*=args.toughness

    #Adjusting all cmc embeddings by cmc_weight
    card_data.loc[:, 'CMC']*=args.cmc
    card_data.loc[:, 'CMC'] = weight_section(card_data.loc[:, 'CMC'],1)
    card_data.loc[:, 'CMC']*=args.cmc
    return card_data

#does normalizing by finding the width of the "section" and the max val in each column and then applying a function for every value in that column
#sqrt(value)/sqrt(max+sqrt(width))
#the hope is to generally end up with a 0-1 range where using square roots means outliers get value but still squished, and the width is double reduced in imact
#generally 0-1 is the largest jump, then 1-2, etc
def weight_section(section: pandas.DataFrame, width):
    if width > 1:
        for column in section:
            
            max = section[column].max()
            if max == 0:
                continue
            section[column] = section[column].apply(lambda x: adjusted_values(x,width,max))
            # oldmax = section[column].max()
            # oldmin = section[column].min()
            # if oldmax != 0:
            #     section[column] = section[column].apply(lambda x: (-1 + (x-oldmin) * ((2)/(oldmax-oldmin))/sqrt(width)))
    else:
        max = section.max()
        section = section.apply(lambda x: adjusted_values(x,width,max))
        # oldmax = section.max()
        # oldmin = section.min()
        # if oldmax != 0:
        #     section = section.apply(lambda x: (-1 + (x-oldmin) * ((2)/(oldmax-oldmin)))/sqrt(width))
    return section

def adjusted_values(value, width, max):
    if value == 0:
        return 0
    return value/(sqrt(max+sqrt(width)))

def card2vec(cards_data):
    """Takes a cards data and converts it into 7 different vectors and then appends them:

    Types line, text box (including type line and name), mana value, rarity, power, toughness, cmc

    adds name at the end
    """

    #applies a lambada of type2vec on the entire column, the rest of this is just dealing w/ data types. result_type = expanded maxes the 1 wide into n wide where n is # of types
    type_vecs = cards_data.loc[:, 'type_line'].to_frame()
    type_vecs = type_vecs.apply(lambda row: type2vec(str(row)), axis='columns', result_type='expand')
    type_vecs.columns = ['Creature', 'Instant', 'Sorcery', 'Enchantment','Artifact', 'Planeswalker', 'Land', 'Legendary']

    mana_vecs = cards_data.loc[:, 'mana_cost'].to_frame()
    mana_vecs = mana_vecs.apply(lambda row: mana2vec(row.values[0]), axis='columns', result_type='expand')
    mana_vecs.columns = ['W','U','B','R','G','C','Generic','X','Phyrexian','Snow_Mana','Pips']

    rarity_vec = cards_data.loc[:, 'rarity'].to_frame()
    rarity_vec = rarity_vec.apply(lambda row: rarity2vec(row.values[0]), axis='columns', result_type='expand')
    rarity_vec.columns = ['Rarity']

    power_vec = cards_data.loc[:, 'power'].to_frame()
    power_vec = power_vec.apply(lambda row: 0 if isinstance(row.values[0], str) or not (row.values[0].is_integer()) else row.values[0], axis='columns', result_type='expand')
    power_vec.name = 'Power'

    toughness_vec = cards_data.loc[:, 'toughness'].to_frame()
    toughness_vec = toughness_vec.apply(lambda row: 0 if isinstance(row.values[0], str) or not (row.values[0].is_integer()) else row.values[0], axis='columns', result_type='expand')
    toughness_vec.name = 'Toughness'

    cmc_vec = cards_data.loc[:, 'cmc'].to_frame()
    cmc_vec = cmc_vec.apply(lambda row: 0 if isinstance(row.values[0], str) or not (row.values[0].is_integer()) else row.values[0], axis='columns', result_type='expand')
    cmc_vec.name = "CMC"

    name_vec = cards_data.loc[:, 'name']
    name_vec.name = 'Name'

    text_vecs = text2vec(cards_data.loc[:, 'name'].map(str).values+ " " + cards_data['type_line'].map(str).values + " "+  cards_data['oracle_text'].map(str).values)
    text_vecs = pandas.DataFrame(np.row_stack(text_vecs))
    text_vecs.columns = ['text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10','text_11','text_12','text_13','text_14','text_15','text_16','text_17','text_18','text_19','text_20','text_21','text_22','text_23','text_24','text_25','text_26','text_27','text_28','text_29','text_30']

    combined = pandas.concat([type_vecs,text_vecs,mana_vecs,rarity_vec,power_vec,toughness_vec,cmc_vec,name_vec], axis=1)
    #print(text_vecs)

    return combined

def text2vec(text):
        
    
    # preproces the documents, and create TaggedDocuments
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                tags=[str(i)]) for i,
                doc in enumerate(text)]

    # train the Doc2vec model
    model = Doc2Vec(vector_size=30,
                    min_count=5, epochs=60)
    model.build_vocab(tagged_data)
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    # get the document vectors
    document_vectors = [model.infer_vector(
        word_tokenize(doc.lower())) for doc in text]

    return document_vectors

def type2vec(type_line):
    """
    Maps type to binary emebeddings in the following order:

    Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, legendary
    """
    # Card types are Creature, Instant, Sorcery, Enchantment,
    # Artifact, Planeswalker, Land, legendary
    vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if "Creature" in type_line:
        vec[0] = 1.5
    if "Instant" in type_line:
        vec[1] = 1.0
    if "Sorcery" in type_line:
        vec[2] = 1.0
    if "Enchantment" in type_line:
        vec[3] = 1.0
    if "Artifact" in type_line:
        vec[4] = 1.0
    if "Planeswalker" in type_line:
        vec[5] = 1.0
    if "Land" in type_line:
        vec[6] = 1.0
    if "Legendary" in type_line:
        vec[7] = 0.6
    vec_magnitude = np.sum(np.pow(vec,2))
    if vec_magnitude == 0:
        return vec
    if vec[6] != 0:
        vec[6]*=2
    return vec / np.sqrt(vec_magnitude)

def mana2vec(mana_cost):

    """
    Converts mana cost to vector

    Embeddings are: W, U, B, R, G, C, Gen, X, Phy, S, Pips
    
    For specific symbols, the corrosponding embedding is increased by 1. Ex: {W} -> [+1,+0,...], with the same for X symbols, pyrexian etc
     
    Pips is the total number of {} in the mana cost, so generic, X, etc are counted as pips.
     
    Generic is the total generic that could be used to cast the spell, Reaper King has Generic = 10 
    """
    if pandas.isnull(mana_cost):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for i in mana_cost:
        
        if "W" == i:
            vec[0] += 1.0
        elif "U" == i:
            vec[1] += 1.0
        elif "B" == i:
            vec[2] += 1.0
        elif "R" == i:
            vec[3] += 1.0
        elif "G" == i:
            vec[4] += 1.0
        elif "C" == i:
            vec[5] += 1.0
        elif i.isdigit():
            vec[6] += float(i)
        elif "X" == i:
            vec[7] += 1.0
        elif "P" in i:
            vec[8] += 1.0
            if "W" in i:
                vec[0] += 1.0
            if "U" == i:
                vec[1] += 1.0
            if "B" == i:
                vec[2] += 1.0
            if "R" == i:
                vec[3] += 1.0
            if "G" == i:
             vec[4] += 1.0
        elif "S" in i:
            vec[9] += 1.0
        elif "{" in i:
            vec[10]+=1.0
    return vec

def rarity2vec(rarity):
    if rarity == 'common':
        vec = [1.0]
    elif rarity == 'uncommon':
        vec = [2.0]
    elif rarity == 'rare':
        vec = [3.0]
    elif rarity == 'mythic':
        vec = [4.0]
    else:
        vec = [0.0]
    return vec

if __name__ == "__main__":
    main()

