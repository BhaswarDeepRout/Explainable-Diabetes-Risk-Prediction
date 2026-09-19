import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "diabetes"
TEST_SIZE = 0.20
RANDOM_STATE = 42

df1 = pd.DataFrame({"X": range(100), "diabetes": [0, 1] * 50})
df_train, df_test = train_test_split(df1, test_size=TEST_SIZE, stratify=df1[TARGET_COLUMN], random_state=RANDOM_STATE)
print("TRAIN INDICES", df_train.index.values[:5])
