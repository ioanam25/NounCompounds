# importing panda library
import csv

import pandas as pd

# readinag given csv file
# and creating dataframe
data = pd.read_csv("compounds.txt");

# storing this dataframe in a csv file
data.to_csv("compounds.csv",
                  index=None)


print(data)

i = 0
print(str(int(40/40)))
# with open("compounds.txt", encoding='utf-8') as compoundsfile:
#     for line in compoundsfile:
#         if i >= 1080:
#             continue
#         print(i, str(int(i / 40)))
#         print(line)
#         currentfile = "batch" + str(int(i / 40)) + ".txt"
#         with open(currentfile, 'a', encoding='utf-8') as file:
#             file.write(line)
#             file.close()
#         i += 1

for i in (0, 26):
    currentfile = "batch" + str(int(i / 40)) + ".txt"
    data = pd.read_csv(currentfile,  encoding='utf-8');

    # storing this dataframe in a csv file
    csvfile = "batch" + str(int(i / 40)) + ".csv"
    data.to_csv(csvfile,
                index=None, encoding='utf-8')