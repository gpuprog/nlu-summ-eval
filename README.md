The evaluation of different Text Summarization algorithms.
P.S. You need a GPU CUDA and a minimum 16Gb of GPU memory.

## How to work

### 1. Create environments.
At first step please create environments.

The "tf1" witch packages:
python=3.6.13
tensorflow=1.2.1
pandas=1.1.5

The "nlu" with packages:
python=3.9.12
datasets==2.6.1
rouge-score==0.1.2
transformers==4.22.2
openai

P.S. Also you shopuld install ipykernel in both environments (it will be requested by Visual Studio Code)

### 2. Setup your OpenAI API key.
Create a file openai-api-key.txt (under the "root" project folder) and push your key into this file.

### 3. Download datasets.
It uses three datasets:
1. [CNN/Daily-Mail](https://github.com/abisee/cnn-dailymail) and [processed files](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail) - "CNN" - download the [finished_files.zip](https://drive.google.com/file/d/0BzQ6rtO2VN95a0c3TlZCWkl3aU0/view?usp=sharing&resourcekey=0-toctC3TNM1vffPCZ7XT0JA) file and extract into "./data-cnn" folder (it will create finished_files subfolder)
2. [Amazon Food reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) - "Amazon" - download and unpack archive.zip direct into "./data-amazon" folder
3. [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset) - "WikiHow" - download the [wikihowSep.csv](https://ucsb.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358) into "./data-wikihow" folder

### 4. Evaluate subprojects independently
Then you can open "lstm", "roberta" and "gpt3" folders with Visual Studio Code (Python extension v2020.12.424452561 recommended - a later extensions can be working incorrect with Python 3.6).
Please read README.md from each of that folder.