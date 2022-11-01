import datasets
import warnings
import pandas as pd
from datasets import Dataset

class Testset:
    def __init__(self, test_data, pred_name, label_name):
        self.data = test_data
        self.pred_name = pred_name
        self.label_name = label_name

def cnnTestset(big=False):
    test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test")

    # Delete for full test
    if not big:
        test_data = test_data.select(1000)

    return Testset(test_data, 'article', 'highlights')

def amazonTestset(big=False):
    df = pd.read_csv("../data-amazon/Reviews.csv")
    df.drop(columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator','HelpfulnessDenominator', 'Score', 'Time'], axis=1, inplace=True)
    df = df.dropna()
    
    test_df = df.sample(frac = (11000 if big else 1000)/len(df), random_state = 25)
    # other_data = df.drop(test_df.index)
    print("Data size:", len(df), "\nTest size:", len(test_df))
    test_data = Dataset.from_pandas(test_df)

    #test_data = Dataset.from_pandas(df[556000:557000])
    #test_data = Dataset.from_pandas(df[556000:556100])

    return Testset(test_data, 'Text', 'Summary')
    
def wikihowTestset(big=False):
    df = pd.read_csv("../data-wikihow/wikihowSep.csv", on_bad_lines='warn')
    df.drop(columns=['overview', 'sectionLabel', 'title'], axis=1, inplace=True)
    df = df.dropna()

    test_df = df.sample(frac = (11000 if big else 1000)/len(df), random_state = 25)

    print("Data size:", len(df), "\nTest size:", len(test_df))
    test_data = Dataset.from_pandas(test_df)

    return Testset(test_data, 'text', 'headline')

def evaluate(model, tokenizer, testset):
    batch_size = 24
    warnings.filterwarnings("ignore")

    # map data correctly
    def generate_summary(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        # cut off at BERT max length 512
        inputs = tokenizer(batch[testset.pred_name], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = model.generate(input_ids, attention_mask=attention_mask)

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred"] = output_str

        return batch

    results = testset.data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=[testset.pred_name])
    pred_str = results["pred"]
    label_str = results[testset.label_name]

    rouge = datasets.load_metric("rouge")
    print("ROUGE-1 SCORE: ", rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid)
    print("ROUGE-2 SCORE: ", rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid)
    print("ROUGE-L SCORE: ", rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid)

    return rouge, pred_str, label_str