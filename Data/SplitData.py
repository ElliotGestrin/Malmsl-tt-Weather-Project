# Shuffles the data and splits it into a test and a training set
import pandas as pd

files = ["MalmslättEarly", "MalmslättLate","MalmslättFull"]

for file in files:
    df = pd.read_csv(file+".csv")
    df = df.sample(frac=1,random_state=1337) # Shuffle the dataset

    N = len(df)
    train = df.iloc[:int(N*0.7),:]
    test = df.iloc[int(N*0.7):,:]

    train.to_csv(file+"_train.csv",index=False)
    test.to_csv(file+"_test.csv",index=False)