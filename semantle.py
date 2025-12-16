import math
import sys
import pandas
import argparse
import numpy as np
import time
import sklearn

parser = argparse.ArgumentParser(prog='1',
                    description='2',
                    epilog='<3')
parser.add_argument("filename")
parser.add_argument("name")
parser.add_argument('-t','--textmul', type=float, default= 2, help= "multiplier for text")
parser.add_argument('-n','--ntextmul', type=float, default= 0.8, help= "multiplier for non-text")
parser.add_argument('-d','--debug', type=bool, default= False, help= "debug")


text_list = ['text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10','text_11','text_12','text_13','text_14','text_15','text_16','text_17','text_18','text_19','text_20','text_21','text_22','text_23','text_24','text_25','text_26','text_27','text_28','text_29','text_30']

def main():
    args = parser.parse_args()
    file = args.filename
    cards = pandas.read_csv(file)
    names = list(cards.loc[:,'Name'].values)
    cards = cards.set_index('Name')
    only_text = cards.loc[:,'text_1' : 'text_30']
    #print(cards.shape[0])
    count = cards.shape[0]
    distances = np.full((count, count),0.0)
   # print(distances)
    #print(cards.values)

    distances_text = pandas.DataFrame(sklearn.metrics.pairwise_distances(only_text.values*args.textmul, only_text.loc[args.name].values.reshape(1,-1)*args.textmul, n_jobs=10))
    cards_no_text = cards.drop(columns=text_list)
    distances_no_text = pandas.DataFrame(sklearn.metrics.pairwise_distances(cards_no_text.values*args.ntextmul, cards_no_text.loc[args.name].values.reshape(1,-1)*args.ntextmul, n_jobs=10))
    
    # with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    
    print(str(pandas.concat([pandas.DataFrame(names),distances_text, distances_no_text],axis=1))+" t_av: "+str(distances_text.mean())+" n_t_av: "+str(distances_no_text.mean()))

    distances = pandas.DataFrame((distances_no_text.iloc[:,0]**2 + distances_text.iloc[:,0]**2).pow(1/2))

    
    distances.index = names
    distances_text.index = names
    distances_no_text.index = names

    distout = distances.iloc[:,0].sort_values(ascending=True)
    print(distout.head(30))
    if args.debug:
        dist_text_out = distances_text.iloc[:,0].sort_values(ascending=True)
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(dist_text_out.head(1000))
        
        dist_no_text_out = distances_no_text.iloc[:,0].sort_values(ascending=True)
        print(dist_no_text_out.head(10))
    save_name = "distances-"+file
    output = pandas.concat([distout, distout.index.to_series()], axis=1)
    print(output)
    output.to_csv(save_name, index=0)
    
            

        


    return 0

def calc_distances(index):


    return 0



if __name__ == "__main__":
    main()