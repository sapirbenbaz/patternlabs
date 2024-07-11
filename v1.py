import re
import os
import random
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# Having the model stored in a class, on which certain actions can be applied
class GeminiModelInstance:
    def __init__(self, api_key, temperature):
        self.instance = self.create_gemini_instance(api_key, temperature)

    # This method creates an instance of Gemini via the SDK and takes two configuration
    # arguments - the API key and temperature
    def create_gemini_instance(self, api_key, temperature):
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
            temperature=temperature,
            google_api_key=api_key,
        )

    # This method invokes the Gemini instance with the prompt received. As we expect an answer
    # between the <answer></answer> brackets, we will keep on invoking the instance as long as
    # there hasn't been a match with the regex requiring an answer of A or B inside the brackets
    def get_model_answer(self, prompt):
        match = None
        while not match:
            response = self.instance.invoke(prompt)
            match = re.search(r"<answer>(A|B)</answer>", response.content)
        return match.group(1)


# Reading via the pandas library the designated json lines file
def load_json_lines(path_or_buf):
    return pd.read_json(path_or_buf=path_or_buf, lines=True)


# Generating the prompt that will be passed to the invoke method
def generate_prompt(question, answer1, answer2):
    return f"""I will give you a question or sentence to complete and two possible answers.
               Please answer either A or B, depending on which answer is better.
               You may write down your reasoning but please write your final answer (either A or B) between the <answer> and </answer> tags. 
               Please try to make sure your answer is correct and as precise as possible.
               The question or sentence is: {question}, the first answer is: {answer1}, the second answer is: {answer2}"""


def main():
    load_dotenv()
    correct_answers = 0
    api_key = os.environ.get("API_KEY")
    data_url = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train.jsonl"
    answers_url = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train-labels.lst"
    gemini_instance = GeminiModelInstance(api_key, 0.1)
    dataset, dataset_answers = load_json_lines(data_url), load_json_lines(answers_url)
    number_of_questions = dataset.shape[0]

    try:
        for _ in range(50):
            # Generating a random index, getting the question, two optional solutions and correct answer
            index = random.randint(0, number_of_questions - 1)
            question, answer1, answer2 = dataset.values[index]
            correct_answer = dataset_answers.values.tolist()[index][0]
            prompt = generate_prompt(question, answer1, answer2)

            # Getting the model's answer and verifying its accuracy
            model_answer = gemini_instance.get_model_answer(prompt)
            if (model_answer == "A" and correct_answer == 0) or (
                model_answer == "B" and correct_answer == 1
            ):
                correct_answers += 1
    except Exception as err:
        print(f"An error has occurred: {err}. Please try again later")

    print(f"Success rate is: {(correct_answers / 50) * 100}")


if __name__ == "__main__":
    main()
