import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

dataset_loc = 'data/UpdatedResumeDataSet.csv'

df = pd.read_csv(dataset_loc)

# df.columns => [Category, Resume]

category_encoder = LabelEncoder()
category_encoder.fit(df['Category'])
print(list(category_encoder.classes_))
df['Category'] = category_encoder.transform(df['Category'])
print(list(df['Category']))