import seaborn as sns
import pandas as pd

def load_penguins():
    df = sns.load_dataset('penguins')
    return df
