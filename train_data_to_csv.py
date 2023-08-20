import os
import pandas as pd

path = "/mnt/d/DVD/zbiory_danych/WOW_0.4bpp/train"

# Iterators over files in respective folders
cover = os.scandir(os.path.join(path, 'cover'))
stego = os.scandir(os.path.join(path, 'stego'))

# Create a list to store data
data = []

for img in cover:
    loc = img.name
    data.append({'file_name': loc, 'label': 0})

for img in stego:
    loc = img.name
    data.append({'file_name': loc, 'label': 1})

# Create a DataFrame from the list
df = pd.DataFrame(data, columns=['file_name', 'label'])

# Save as csv file
df.to_csv('train_data.csv', header=None, index=False)
