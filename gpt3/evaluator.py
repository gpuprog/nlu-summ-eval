import datasets
import warnings
import pandas as pd
from datasets import Dataset
import openai
from tqdm import tqdm

number = 100

class Testset:
    def __init__(self, test_data, pred_name, label_name):
        self.data = test_data
        self.pred_name = pred_name
        self.label_name = label_name

def cnnTestset():
    test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")

    # Delete for full test
    test_data = test_data.select(range(number))

    return Testset(test_data, 'article', 'highlights')

def amazonTestset():
    df = pd.read_csv("../data-amazon/Reviews.csv", quotechar='"', quoting=1)
    df.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator','HelpfulnessDenominator', 'Score', 'Time'], axis=1, inplace=True)
    df = df.dropna()
    
    test_df = df.sample(frac = number/len(df), random_state = 25)
    # other_data = df.drop(test_df.index)
    print("Data size:", len(df), "\nTest size:", len(test_df))
    test_data = Dataset.from_pandas(test_df)

    return Testset(test_data, 'Text', 'Summary')
    
def wikihowTestset():
    df = pd.read_csv("../data-wikihow/wikihowSep.csv", on_bad_lines='warn', quotechar='"', quoting=1)
    df.drop(columns=['overview', 'sectionLabel', 'title'], axis=1, inplace=True)
    df = df.dropna()

    test_df = df.sample(frac = number/len(df), random_state = 25)

    print("Data size:", len(df), "\nTest size:", len(test_df))
    test_data = Dataset.from_pandas(test_df)

    return Testset(test_data, 'text', 'headline')

def evaluate(testset):
    with open('../openai-api-key.txt') as f:
        openai_api_key = f.readline()

    label_str = testset.data[testset.label_name]

    def ask(text):
        openai.api_key = openai_api_key
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt =  text,
            temperature = 0.7,
            top_p = 1,
            max_tokens = 128,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        return response.choices[0].text

    pred_str = []
    #tqdm(df.iterrows(), total=df.shape[0]):
    for text in tqdm(testset.data):
        text = text[testset.pred_name]
        resp = ask(text + "\nTL;DR")
        pred_str.append(resp)

    rouge = datasets.load_metric("rouge")
    print("ROUGE-1 SCORE: ", rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid)
    print("ROUGE-2 SCORE: ", rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid)
    print("ROUGE-L SCORE: ", rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid)

    return rouge, pred_str, label_str