import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train_df = pd.read_csv("./input/train.csv")
    print(train_df)
    train_df, test_df = train_test_split(train_df, test_size=0.1, shuffle=False)
    train_df.to_csv("./input/train_split.csv", index=False)
    test_df.to_csv("./input/test_split.csv", index=False)
