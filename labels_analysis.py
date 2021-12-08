# import required module
import os
import csv
import pandas as pd

directory = 'labels'

freq_label = {}
hits = {}
label_agrees = {}
confused = {}

count = 0
agreed = 0
for filename in os.listdir(directory): # for each batch
    data = pd.read_csv('labels' + '\\' + filename)
    for label in data['Answer.category.label']: # for each label in the batch
        # print(label)
        count += 1
        if label not in freq_label:
            freq_label[label] = 0
        freq_label[label] += 1

    for worker in data['WorkerId']: # for each worker
        if worker not in hits:
            hits[worker] = 0
        hits[worker] += 1

    for i in range(0, 80, 2):
        if data['Answer.category.label'][i] == data['Answer.category.label'][i + 1]:
            agreed += 1
            if data['Answer.category.label'][i] not in label_agrees:
                label_agrees[data['Answer.category.label'][i]] = 0
            label_agrees[data['Answer.category.label'][i]] += 1
        # print(data['Input.C1'][i], data['Answer.category.label'][i])
        else:
            x = data['Answer.category.label'][i]
            y = data['Answer.category.label'][i + 1]
            if x > y:
                x, y = y, x
            if (x, y) not in confused:
                confused[(x, y)] = 0
            confused[(x, y)] += 1

print(confused)

file_agrees = open('category agreements', 'w', encoding='utf-8')
file_agrees.write("Category + how many times people agreed on that category for a compound \n")
file_agrees.write("Total number of compounds that received the same label from both annotators " + str(agreed) + '\n')
for key, val in label_agrees.items():
    file_agrees.write(key + ' ' + str(val) + '\n')

file_hits = open('workers_and_hits', 'w', encoding='utf-8')
file_hits.write("worker id and how many compounds they annotated\n")
for key, val in hits.items():
    file_hits.write(key + ' ' + str(val) + '\n')

sorted_freq_labels = {}

sorted_freq_labels = sorted(freq_label.items(), key=lambda x: x[1], reverse=True)

file = open('frequency_of_labels.txt', 'w', encoding='utf-8')
file.write("category + how many labels are in that category\n")
for i in sorted_freq_labels:
    file.write(str(i) + '\n')


