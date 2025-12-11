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


def main():
    args = parser.parse_args()
    file = args.filename
    cards = pandas.read_csv(file)
    names = list(cards.loc[:,'Name'].values)
    cards = cards.set_index('Name')
    #print(cards.shape[0])
    count = cards.shape[0]
    distances = np.full((count, count),0.0)
   # print(distances)
    #print(cards.values)

    t = time.time()

    distances = sklearn.metrics.pairwise_distances(cards.values, cards.values, n_jobs=10)


    #change this to sklearn.metrics.parwise_distances
    # for outer in range(count):
    #     c1_vals = cards.iloc[outer].values
    #     print("inner: ",outer ," at time" ,(time.time() - t) * 1e3, "ms")
    #     for inner in range(count):
            
    #         if inner == outer:
    #             distances[inner][outer] = 0.0
    #         else:
    #             c2_vals = cards.iloc[inner].values
    #             sum = 0.0
    #             for iter in range(len(c1_vals)):
    #                 sum+= (c1_vals[iter]-c2_vals[iter])**2.0
    #             distances[inner][outer]=math.sqrt(sum)
    print(distances)

    final = pandas.DataFrame(distances)
    final.columns = names
    final.index = names
    print(final)
    
    test = final.loc[:,args.name]
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(test.sort_values(ascending=False))

    save_name = "distances-"+file
    cards.to_csv(save_name, index=0)
    
            

        


    return 0

def calc_distances(index):


    return 0



if __name__ == "__main__":
    main()