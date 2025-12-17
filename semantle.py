import math
import sys
import pandas
import argparse
import numpy as np
import time
import sklearn

DEFAULT_TEXTMUL = 1
DEFAULT_NTEXTMUL = 1

parser = argparse.ArgumentParser(prog='1',
                    description='2',
                    epilog='<3')
parser.add_argument("filename")
parser.add_argument("name")
parser.add_argument('-t','--textmul', type=float, default= DEFAULT_TEXTMUL, help= "multiplier for text")
parser.add_argument('-n','--ntextmul', type=float, default= DEFAULT_NTEXTMUL, help= "multiplier for non-text")
parser.add_argument('-s','--savefile', type=bool, default= True, help= "debug")
parser.add_argument('-d','--debug', type=bool, default= False, help= "debug")


text_list = ['text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10','text_11','text_12','text_13','text_14','text_15','text_16','text_17','text_18','text_19','text_20','text_21','text_22','text_23','text_24','text_25','text_26','text_27','text_28','text_29','text_30']

def main():
    
    args = parser.parse_args()
    semantle(args.filename, args.name, args.textmul, args.ntextmul, args.savefile, args.debug)

    return 0

def semantle(file_name, cardname ,textmul = DEFAULT_TEXTMUL, ntextmul = DEFAULT_NTEXTMUL, savefile = False, debug = False):
    print(cardname)
    v_prefix = "vector_"
    file_name
    cards = pandas.read_csv(v_prefix+file_name)
    names = list(cards.loc[:,'Name'].values)
    cards = cards.set_index('Name')
    only_text = cards.loc[:,'text_1' : 'text_30']
    #print(cards.shape[0])
    count = cards.shape[0]
    distances = np.full((count, count),0.0)
   # print(distances)
    #print(cards.values)

    distances_text = pandas.DataFrame(sklearn.metrics.pairwise_distances(only_text.values*textmul, only_text.loc[cardname].values.reshape(1,-1)*textmul, n_jobs=10))
    cards_no_text = cards.drop(columns=text_list)
    distances_no_text = pandas.DataFrame(sklearn.metrics.pairwise_distances(cards_no_text.values*ntextmul, cards_no_text.loc[cardname].values.reshape(1,-1)*ntextmul, n_jobs=10))
    
    # with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    
    #print(str(pandas.concat([pandas.DataFrame(names),distances_text, distances_no_text],axis=1))+" t_av: "+str(distances_text.mean())+" n_t_av: "+str(distances_no_text.mean()))

    distances = pandas.DataFrame((distances_no_text.iloc[:,0]**2 + distances_text.iloc[:,0]**2).pow(1/2))

    
    distances.index = names
    distances_text.index = names
    distances_no_text.index = names

    distout = distances.iloc[:,0].sort_values(ascending=True)
    #print(distout.head(30))
    if debug:
        dist_text_out = distances_text.iloc[:,0].sort_values(ascending=True)
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(dist_text_out.head(1000))
        
        dist_no_text_out = distances_no_text.iloc[:,0].sort_values(ascending=True)
        print(dist_no_text_out.head(10))
    save_name = "distances-"+file_name
    output = pandas.concat([distout, distout.index.to_series()], axis=1)
    output.columns = ["Distance","Card_names"]
    #print(output)
    if savefile:
        output.to_csv(save_name, index=0)

    return distout




if __name__ == "__main__":
    main()