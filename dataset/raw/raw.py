import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/solution-seeker-as/manywells/data/manywells-sol-1.zip")

print(df)