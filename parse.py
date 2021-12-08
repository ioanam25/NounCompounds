from conllu import parse_incr
import matplotlib.pyplot as plt
import rowordnet as rwn

wn = rwn.RoWordNet()

data = open("UD_Romanian-RRT/ro_rrt-ud-train.conllu", "r", encoding="utf-8")

sentences = list(parse_incr(data))
nounCompounds_gen = []
nounCompounds_acc = []
nounCompounds_npn = []
nounCompounds_npn_elem = []
nounCompounds_gen_elem = []
count = 0
NN = 0 # noun noun
NPN = 0 # noun preposition noun
genitive = 0 # genitive marked noun noun
acc = 0 # acuzativ / nominativ ?
cases = 0
total_tokens = 0
nouns_in_compounds = 0
total_nouns = 0
diff_nouns = 0
lemmas = []
prepositions = {}
noun_freq = {}  # freq, heads, modifiers, freq rowordnet
heads = {}
modifiers = {}
heads_prep = {}
modifiers_prep = {}

file = open('nouns.txt', 'w', encoding='utf-8')
file_freq = open('freq.txt', 'w', encoding='utf-8')

# file.write(s)

for sentence in sentences:
    for token in sentence:
        print(token)
        total_tokens += 1
        # get next two token if they exist
        if sentence.index(token) < len(sentence) - 1:
            next_token = sentence[sentence.index(token) + 1]
            if (sentence.index(token) + 1) < len(sentence) - 1:
                next2_token = sentence[sentence.index(next_token) + 1]

        # check nouns
        if token['upos'] == "NOUN":
            total_nouns += 1

            if token['lemma'] not in lemmas:
                lemmas.append(token['lemma'])
                diff_nouns += 1

            # Noun-noun compound
            if next_token['upos'] == "NOUN":
                NN += 1

                if next_token['feats'] is not None and 'Case' in next_token['feats'].keys():

                    if 'Case' in next_token['feats'].keys():
                        if next_token['feats']['Case'] == "Dat,Gen":
                            # print(token, next_token)
                            if token['form'] + "_" + next_token['form'] not in nounCompounds_gen_elem:
                                nounCompounds_gen.append((token['form'] + "_" + next_token['form'], token, next_token))
                                genitive += 1
                                nounCompounds_gen_elem.append(token['form'] + "_" + next_token['form'])

                                if token['lemma'] not in noun_freq.keys():
                                    noun_freq[token['lemma']] = [0, 0, 0, 0]
                                    synset_ids = wn.synsets(literal=token['lemma'])
                                    noun_freq[token['lemma']][3] = len(synset_ids)

                                # frequencies of compounds in NN compounds
                                nouns_in_compounds += 2
                                noun_freq[token['lemma']][0] += 1
                                noun_freq[token['lemma']][1] += 1 # head

                                if next_token['lemma'] not in noun_freq.keys():
                                    noun_freq[next_token['lemma']] = [0, 0, 0, 0]
                                    synset_ids = wn.synsets(literal=next_token['lemma'])
                                    noun_freq[next_token['lemma']][3] = len(synset_ids)

                                noun_freq[next_token['lemma']][0] += 1
                                noun_freq[next_token['lemma']][2] += 1 # modifier

                                if (token['form']) not in heads.keys():
                                    heads[token['form']] = []
                                heads[token['form']].append(next_token['form'])

                                if (next_token['form']) not in modifiers.keys():
                                    modifiers[next_token['form']] = []
                                modifiers[next_token['form']].append(token['form'])

                        else:
                            if (token['form'] + "_" + next_token['form'], token, next_token) not in nounCompounds_acc:
                                nounCompounds_acc.append((token['form'] + "_" + next_token['form'], token, next_token))
                                acc += 1

            elif next_token['upos'] == "ADP" and next2_token['upos'] == "NOUN":
                if token['lemma'] not in noun_freq.keys():
                    noun_freq[token['lemma']] = [0, 0, 0, 0] # freq, heads, modifiers, fq in rowordnet
                    synset_ids = wn.synsets(literal=token['lemma'])
                    noun_freq[token['lemma']][3] = len(synset_ids)

                if token['form'] + "_" + next_token['form'] + "_" + next2_token['form'] not in nounCompounds_npn_elem:
                    nounCompounds_npn.append((token['form'] + "_" + next_token['form'] + "_" + next2_token['form'], token,
                                         next_token, next2_token))
                    nounCompounds_npn_elem.append(token['form'] + "_" + next_token['form'] + "_" + next2_token['form'])

                    # frequencies of nouns in NPN compounds
                    nouns_in_compounds += 2
                    noun_freq[token['lemma']][0] += 1
                    noun_freq[token['lemma']][1] += 1  # head

                    if next2_token['lemma'] not in noun_freq.keys():
                        noun_freq[next2_token['lemma']] = [0, 0, 0, 0]
                        synset_ids = wn.synsets(literal=next2_token['lemma'])

                        noun_freq[next2_token['lemma']][3] = len(synset_ids)

                    noun_freq[next2_token['lemma']][0] += 1
                    noun_freq[next2_token['lemma']][2] += 1  # modifier

                    # list of prepositions and the compounds they help form
                    if next_token['form'] not in prepositions.keys():
                        prepositions[next_token['form']] = []
                    if token['form'] + "_" + next_token['form'] + "_" + next2_token['form'] not in prepositions[next_token['form']]:
                        prepositions[next_token['form']].append(token['form'] + "_" + next_token['form'] + "_" + next2_token['form'])
                        NPN += 1

                    if (token['form'] + "_" + next_token['form']) not in heads.keys():
                        heads[token['form'] + "_" + next_token['form']] = []
                        heads_prep[(token['form'], next_token['form'])] = 0
                    heads[token['form'] + "_" + next_token['form']].append(next2_token['form'])
                    heads_prep[(token['form'], next_token['form'])] += 1

                    if (next_token['form'] + "_" + next2_token['form']) not in modifiers.keys():
                        modifiers[next2_token['form']] = []
                        modifiers_prep[(next_token['form'], next2_token['form'])] = 0
                    modifiers[next2_token['form']].append(token['form'] + "_" + next_token['form'])
                    modifiers_prep[(next_token['form'], next2_token['form'])] += 1

sort_heads_prep= sorted(heads_prep.items(), key=lambda x: x[1], reverse=True)
sort_modifiers_prep = sorted(modifiers_prep.items(), key=lambda x: x[1], reverse=True)

# for key, val in prepositions.items():
#     file.write(str(key) + " " + str(len(val)) + "\n")
#     for i in val:
#         file.write(str(i) + ", ")
#     file.write("\n")

# markdict={"Tom":67, "Tina": 54, "Akbar": 87, "Kane": 43, "Divya":73}
# marklist=list(markdict.items())
# print(marklist)
# markdict = {"Tom":67, "Tina": 54, "Akbar": 87, "Kane": 43, "Divya":73}
# l = len(marklist)
# for i in range(l-1):
#     for j in range(i+1,l):
#         if marklist[i][1]>marklist[j][1]:
#             t=marklist[i]
#             marklist[i]=marklist[j]
#             marklist[j]=t
#     sortdict=dict(marklist)
#     print(sortdict)

headslist = list(heads.items())
l = len(headslist)
for i in range(l - 1):
   for j in range(i + 1, l):
       # print(headslist[j][1])
       if len(headslist[i][1]) < len(headslist[j][1]):
            aux = headslist[i]
            headslist[i] = headslist[j]
            headslist[j] = aux
heads = dict(headslist)

file_table = open('table.txt', 'w', encoding='utf-8')
# file_compounds_list = open('compoundslist.txt', 'w', encoding='utf-8')
for key, value in heads.items():
    # print(key)
    file_table.write(key + '\n')
    for i in range(len(value) - 1):
        for j in range(i + 1,  len(value)):
            if len(modifiers[value[i]]) < len(modifiers[value[j]]):
                aux = value[i]
                value[i] = value[j]
                value[j] = aux
    for elem in value:
        # print(elem, len(modifiers[elem]))
        file_table.write(elem + " " + str(len(modifiers[elem])) + '\n')
    file_table.write('\n')
    # file_compounds_list.write(key + "_" + value[0] + '\n')

modifierslist = list(modifiers.items())
l = len(modifierslist)
for i in range(l - 1):
   for j in range(i + 1, l):
       # print(headslist[j][1])
       if len(modifierslist[i][1]) < len(modifierslist[j][1]):
            aux = modifierslist[i]
            modifierslist[i] = modifierslist[j]
            modifierslist[j] = aux
modifiers = dict(modifierslist)
# print(sort_heads)

file_h = open('heads.txt', 'w', encoding='utf-8')
file_h.write("Head + different modifiers \n")
for key, val in heads.items():
    file_h.write(str(key) + " " + str(val) + '\n')

file_m = open('modifiers.txt', 'w', encoding='utf-8')
file_m.write("Modifier + different heads \n")
for key, val in modifiers.items():
    file_m.write(str(key) + " " + str(val) + '\n')


# print(modifiers)
# inwn = 0
# for word in nounCompounds_gen:
#     if wn.synsets(literal=word[0]):
#         print(word[0])
#         inwn += 1
#
# for word in nounCompounds_npn:
#     if wn.synsets(literal=word[0]):
#         print(word[0])
#         inwn += 1
# print(inwn)


sort_noun_freq = sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)
freq_corpus = []
freq_rowordnet = []
file_freq.write("frequency, head, modifier, frequency in rowordnet \n")
for i in sort_noun_freq:
    file_freq.write(str(i) + '\n')
    freq_corpus.append(i[1][0])
    freq_rowordnet.append(i[1][1])

plt.plot(freq_corpus, freq_rowordnet)
plt.xlabel("Frequency of nouns in corpus")
plt.ylabel("Frequency of nouns in RoWordNet")
plt.title("Frequency in corpus vs RoWordnet")
plt.savefig("corpusvsrowordnet.svg")
plt.show()

freq = []
for item in sort_noun_freq:
    freqs = item[1]
    fr = freqs[0]
    freq.append(fr)

keys = []
# print(len(noun_freq.keys()))
keys = list(range(0, len(noun_freq.keys())))

plt.plot(keys, freq)
plt.xlabel("Most to least frequent nouns")
plt.ylabel("Frequencies")
plt.title("Distribution of noun frequencies")
plt.savefig("nounfrequencies.svg")
plt.show()

file.write("total tokens: " + str(total_tokens) + "\n")
file.write("total nouns: " + str(total_nouns) + "\n")
file.write("types of nouns: " + str(diff_nouns) + "\n")
file.write("NN case acc: " + str(acc) + "\n")
for compound in nounCompounds_acc:
    file.write(compound[0] + ", ")
file.write("\n")
file.write("NN case dat/gen: " + str(genitive) + "\n")
for compound in nounCompounds_gen:
    file.write(compound[0] + ", ")
file.write("\n")
file.write("total NPN: " + str(NPN) + "\n")
for key, val in prepositions.items():
    file.write(str(key) + " " + str(len(val)) + "\n")
    for i in val:
        file.write(str(i) + ", ")
    file.write("\n")

# word = 'timp'
# print("Search for all noun synsets that contain word/literal '{}'".format(word))
# synset_ids = wn.synsets(literal=word)
# for synset_id in synset_ids:
#     print(wn.synset(synset_id))

# synset_id = wn.synsets("tren")[2] # select the third synset from all synsets containing word "tren"
# print("\nPrint all outbound relations of {}".format(wn.synset(synset_id)))
# outbound_relations = wn.outbound_relations(synset_id)
# for outbound_relation in outbound_relations:
#     target_synset_id = outbound_relation[0]
#     relation = outbound_relation[1]
#     print("\tRelation [{}] to synset {}".format(relation,wn.synset(target_synset_id)))
#
# synset_id = wn.synsets("tren")[0] # select the second synset from all synsets containing word "tren"
# print("\nPrint all outbound relations of {}".format(wn.synset(synset_id)))
# outbound_relations = wn.outbound_relations(synset_id)
# for outbound_relation in outbound_relations:
#     target_synset_id = outbound_relation[0]
#     relation = outbound_relation[1]
#     print("\tRelation [{}] to synset {}".format(relation,wn.synset(target_synset_id)))