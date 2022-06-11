# Auto-generate essay for IELST Writing task 2

IELST is the name of an English test, in which examinees need to complete four component tests corresponding with four fundamental skills of Listening, Reading, Speaking, and Writing. Amongst these skills, Writing has been widly considered as a troublesome task.

In this repo, I tried to teach an AI how to generate an IELST Writing task 2 essay given a question. I used GPT2 with some prompt tricks to accomplish the task.

## Methods

The dataset consists of pairs of question and answer crawled from [this website](https://www.ielts-mentor.com/writing-sample/writing-task-2). For every pair of question and answer, for examples `"What is the meaning of life?"` and `"Life is constant suffering."`, the model input is created by concatenating the question and the answer, after prefixing them with their corresponding tags, as followed:
```
"<|question|>What is the meaning of life?<|answer|>Life is constant suffering.<|endoftext|>"
```
In total I was only able to gather 2415 sammples, splited into train and val set with the ratio of 0.9:0.1.
This number of samples is definitely low, which is reflected in the modest quality of the model's output.

The pretrained [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel) model was downloaded from the HuggingFace library.

## Usage

### Installing requirements:
After cloning this repo, install the requirements by running:
```
pip install -r requirements.txt
```
### Finetuning GPT2...
The hyper-parameters for training are stored in [`config/config.yaml`](config/config.yaml). You can left it as is, or write over your own parameters, or create a new `.yaml` file in the same directory.

To start finetuning GPT2, run:
```
python train.py --config=<path/to/config>
```
or simply just:
```
python train.py
```
if you don't create any new config file.

### ...Or downloading the trained model
If you want to skip the training step, you can download the trained model from [here](https://drive.google.com/file/d/1SFEpNQLot3amIjlcuXljzYtoFLttQEVb/view?usp=sharing) and extract right in the current working directory.

### Generating essays
In the [`demo/question.txt`](demo/question.txt) file, enter the questions you want the model to answer. You can add more than one question, and separate the questions using `"\n###\n"`. Finally, run:
```
python generate.py --model_checkpoint=<path/to/model_checkpoing>
```
or just:
```
python generate.py
```
if you chose to download the trained model. The model's answers will be written into [`demo/answer.txt`](demo/answer.txt).

## Personal notes:
Overall, the model can mostly generate coherent short answers, but its output is really bad in terms of essays. This could be due to the shortage of data, which I might increase in the future. 