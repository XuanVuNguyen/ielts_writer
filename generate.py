import os
import argparse
from typing import Union, List
from transformers import TextGenerationPipeline, set_seed
from model.data import IelstDataset
from model.model import GPT2Lightning

def setup_arg_parser():
    parser = argparse.ArgumentParser('Generate')
    parser.add_argument("--model_checkpoint", type=str, 
                        default="logs/test/version_0/checkpoints/last.ckpt")
    parser.add_argument("--input_text", type=str, 
                        default="demo/question.txt")
    parser.add_argument("--set_seed", action="store_true")
    return parser

class TextGenerator:
    tokenizer = IelstDataset.tokenizer
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = GPT2Lightning.load_from_checkpoint(model_path)
        self.q_tag, self.a_tag = self.tokenizer.additional_special_tokens
        self.pipeline = TextGenerationPipeline(self.model.gpt2, IelstDataset.tokenizer)

    def prepare_prompt(self, input_text: Union[str, List[str]]):
        if isinstance(input_text, str):
            input_text = [input_text]
        return [self.q_tag + text.strip() + self.a_tag for text in input_text]
    
    def generate(self, input_text: Union[str, List[str]]):
        input_text = self.prepare_prompt(input_text)
        outputs = self.pipeline(input_text, return_full_text=False, max_length=700)
        return [output[0]["generated_text"] for output in outputs]

def main(args):
    input_file = args.input_text
    directory = os.path.dirname(input_file)
    model_path = args.model_checkpoint
    with open(input_file, "r") as file:
        input_text = file.read()
    input_text = input_text.split("\n###\n")
    generator = TextGenerator(model_path)

    if args.set_seed:
        set_seed(0)
    output_text = generator.generate(input_text)
    output_text = "\n###\n".join(output_text)
    with open(os.path.join(directory, "answer.txt"), "w") as file:
        file.write(output_text)

if __name__=="__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    main(args)
