import json

from config import *


def sentence_accuracy(outputs, std_outputs):
    total = len(std_outputs)
    correct = 0

    for output, std_output in zip(outputs, std_outputs):
        correct += int(output == std_output)
    
    return correct / total

def word_accuracy(outputs, std_outputs):
    total = 0
    correct = 0

    for output, std_output in zip(outputs, std_outputs):
        total += len(std_output)
        for token, std_token in zip(output, std_output):
            correct += int(token == std_token)
    
    return correct / total


def eval():
    for (std_output_file_path, std_output_format), (output_file_path, output_format) in zip(std_output_path, output_path):
        with open(output_file_path, "r", encoding=output_format) as output_file:
            outputs = [line.rstrip() for line in output_file.readlines()]

        with open(std_output_file_path, "r", encoding=std_output_format) as std_output_file:
            std_outputs = [line.rstrip() for line in std_output_file.readlines()]

        return sentence_accuracy(outputs=outputs, std_outputs=std_outputs), word_accuracy(outputs=outputs, std_outputs=std_outputs)

if __name__ == "__main__":
    sentence_acc, word_acc = eval()
    print("sentence accuracy :", sentence_acc)
    print("word accuracy :", word_acc)
