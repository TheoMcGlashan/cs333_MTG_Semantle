Elizabeth Bennet
elizabethbennet
Online

KondoKana — 11/25/25, 4:24 PM
then ima stick the writeup I was working on in the google doc and we can decide if that matches how we want to do things
nvm the doc is just the progress report
here
```Card2Vec:
Function takes in a card and processes it into several vectors using part2vec functions
Combines vectors into final vector representation of cards using "transformation" function
Transformation function initial -> appending vectors

We are not handling, schemes, planes, or un-sets
Expand
message.txt
3 KB
᲼᲼ — 11/25/25, 4:26 PM
yeah we've got most of this figured out, will need expanding upon
once Theo pushes
Elizabeth Bennet

 — 11/25/25, 5:10 PM
It’s pushed it’s just in my branch
Check out the theo branch
Or acrually
I’ll just merge it
But in general you should do coding in your branch and then merge branches in GitHub
᲼᲼ — 11/25/25, 5:11 PM
ew
fuck that
Elizabeth Bennet

 — 11/25/25, 5:13 PM
trust me
᲼᲼ — 11/25/25, 5:13 PM
no
Elizabeth Bennet

 — 11/25/25, 5:13 PM
this is how they specifically instructed us to do groupwork in other cs classes
᲼᲼ — 11/25/25, 5:13 PM
that's cool
Elizabeth Bennet

 — 11/25/25, 5:13 PM
this may very likely save us hours of worj
work
ive seen merge conflicts take whole groups days to fix because they weren't careful with their branches
just write code, push to github, go on github to your branch, and try to merge with main
᲼᲼ — 11/25/25, 5:15 PM
I mean if you wanna teach me how to actually use branches I'll learn, but I won't be happy about it
this'll be a first
Elizabeth Bennet

 — 11/25/25, 5:20 PM
its quite easy, if I see you in person I will
for now it should be fine just to do very frequent commits to main
all you do for a branch though is git checkout -b "name of branch"
and then the first time you push itll say there's no upstream branch because you locally created the branch, but I think the error message tells you the command to run to make a branch in the github repo that matches your local one
᲼᲼ — Yesterday at 1:00 AM
what kind of scaling do we wanna do? I feel like min-max would be best, because we're figuring out distance semantically between cards
᲼᲼ — Yesterday at 1:12 AM
Things will be much easier if we convert the vectors into a csv file, any objections to me doing that once we have the data?
turn the categories into columns
since yall are asleep i'm packing up for tonight
will return tomorrow
Elizabeth Bennet

 — Yesterday at 7:23 AM
What values would we be scaling?
The only attribute values are images, I don’t think there’s a useful way to normalize those
᲼᲼ — Yesterday at 11:34 AM
Wrong project
᲼᲼ — Yesterday at 11:56 AM
Think I misunderstood purpose of code, you good I’m still tinkering
᲼᲼ — Yesterday at 2:36 PM
Yo, Theo, I got questions about the doc2text
can you call real quick?
figured it out
Elizabeth Bennet

 — Yesterday at 3:03 PM
Oh ok good I’m away at the moment
I’ll be back pretty late tonight
᲼᲼ — Yesterday at 3:03 PM
ditto, and you good
I've almost got card2vec working
minor hiccups on normalization but otherwise everything is producing outputs as it should
᲼᲼ — Yesterday at 3:29 PM
Theo I'm not allowed to push to the repository
Elizabeth Bennet

 — Yesterday at 3:48 PM
Great
What’s your GitHub
I can do it in a bit
᲼᲼ — Yesterday at 4:00 PM
Aaryavak
᲼᲼ — 1:28 PM
did you give me push power?
Elizabeth Bennet

 — 1:29 PM
oh freak i forgot my bad
ok i just did
᲼᲼ — 1:29 PM
cool
Elizabeth Bennet

 — 1:29 PM
wait wrong project
but you have that one too
᲼᲼ — 1:30 PM
cool
okay
thanks
Elizabeth Bennet

 — 1:30 PM
ok now you have both
peter send your github as well ill add yoy
᲼᲼ — 1:32 PM
it wants me to make and push to a fork
Elizabeth Bennet

 — 1:33 PM
hmmm I don't know how that works
᲼᲼ — 1:34 PM
okay I have a proposal for you Theo
Elizabeth Bennet

 — 1:34 PM
maybe you can just push and then combine in github?
᲼᲼ — 1:34 PM
I send the contents of my main file here

And you copy and paste it into your main file

and then you push it
Elizabeth Bennet

 — 1:35 PM
i mean sure
we should figure this out for the future though
what does it say specifically?
᲼᲼ — 1:35 PM
yes
but for now
I have a bug I can't figure out and I need other people to stare at it
Elizabeth Bennet

 — 1:35 PM
ok
᲼᲼ — 1:35 PM

import pandas
import sys
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
from sklearn.preprocessing import normalize
import numpy as np

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
        print(index,row)
        card_vec = card2vec(row)
        quit()

        # want to write card_vec to a file alongside the card's name as an identifier.
        # need to contend with writing 30,000 vectors to a file.

def card2vec(card_data):
    type_vec = type2vec(card_data['type_line'])
    text_vec = text2vec(card_data['name'] + ": " + card_data['oracle_text'])
    colors_vec = colors2vec(card_data['colors'])
    rarity_vec = rarity2vec(card_data['rarity'])
    if card_data['power'].is_integer():
        power_vec = card_data['power']
    else:
        power_vec = [0]
    if card_data['cmc'].is_integer():
        cmc_vec = [card_data['cmc']]
    else:
        cmc_vec = [0]
    if card_data['toughness'].is_integer():
        toughness_vec = card_data['toughness']
    else:
        toughness_vec = [0]
    
    
    type_vec = list(map(lambda x: x * type_weight, type_vec))               #type_weight * type_vec   #.normalize()
    text_vec = list(map(lambda x: x * text_weight, text_vec))
    for i in range(len(text_vec)):
        text_vec[i] = text_vec[i].item()               
    colors_vec = list(map(lambda x: x * colors_weight, colors_vec))         #colors_weight * colors_vec #.normalize()
    rarity_vec = list(map(lambda x: x * rarity_weight, rarity_vec))         #rarity_weight * rarity_vec
    power_vec = list(map(lambda x: x * power_toughness_weight, power_vec))  #power_toughness_weight
    toughness_vec = list(map(lambda x: x * power_toughness_weight, toughness_vec))
    cmc_vec = list(map(lambda x: x * cmc_weight, cmc_vec))
    card_vector = [type_vec, text_vec, colors_vec, rarity_vec, power_vec, toughness_vec, cmc_vec]
    card_vector = np.array(card_vector, dtype=object)
    print(card_vector)
    card_vector = normalize(card_vector, copy=False, return_norm=True)
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
... (53 lines left)
Collapse
message.txt
5 KB
Elizabeth Bennet

 — 1:36 PM
ok im working on something else at the moment but ill see if i can find the bug soon
᲼᲼ — 1:36 PM
that's fine, did you push it?
Elizabeth Bennet

 — 1:40 PM
Im combining them now, Im a little confused about some of the changes
do the lines that map multiplication of the weights to all elements in the vectors do anything different than just multiplying the weight by the vector?
᲼᲼ — 1:41 PM
they do not

But your syntax flat-out was wrong, it didn't work
Elizabeth Bennet

 — 1:41 PM
oh I see python doesnt just have .normalize()
᲼᲼ — 1:41 PM
code never got far enough to those cause it kept erroring out earlier
correct
﻿
import pandas
import sys
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
from sklearn.preprocessing import normalize
import numpy as np

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
        print(index,row)
        card_vec = card2vec(row)
        quit()

        # want to write card_vec to a file alongside the card's name as an identifier.
        # need to contend with writing 30,000 vectors to a file.

def card2vec(card_data):
    type_vec = type2vec(card_data['type_line'])
    text_vec = text2vec(card_data['name'] + ": " + card_data['oracle_text'])
    colors_vec = colors2vec(card_data['colors'])
    rarity_vec = rarity2vec(card_data['rarity'])
    if card_data['power'].is_integer():
        power_vec = card_data['power']
    else:
        power_vec = [0]
    if card_data['cmc'].is_integer():
        cmc_vec = [card_data['cmc']]
    else:
        cmc_vec = [0]
    if card_data['toughness'].is_integer():
        toughness_vec = card_data['toughness']
    else:
        toughness_vec = [0]
    
    
    type_vec = list(map(lambda x: x * type_weight, type_vec))               #type_weight * type_vec   #.normalize()
    text_vec = list(map(lambda x: x * text_weight, text_vec))
    for i in range(len(text_vec)):
        text_vec[i] = text_vec[i].item()               
    colors_vec = list(map(lambda x: x * colors_weight, colors_vec))         #colors_weight * colors_vec #.normalize()
    rarity_vec = list(map(lambda x: x * rarity_weight, rarity_vec))         #rarity_weight * rarity_vec
    power_vec = list(map(lambda x: x * power_toughness_weight, power_vec))  #power_toughness_weight
    toughness_vec = list(map(lambda x: x * power_toughness_weight, toughness_vec))
    cmc_vec = list(map(lambda x: x * cmc_weight, cmc_vec))
    card_vector = [type_vec, text_vec, colors_vec, rarity_vec, power_vec, toughness_vec, cmc_vec]
    card_vector = np.array(card_vector, dtype=object)
    print(card_vector)
    card_vector = normalize(card_vector, copy=False, return_norm=True)
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
message.txt
5 KB
