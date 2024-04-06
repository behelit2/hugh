################################################################
# HUGH, a chatbot based on the Microsoft Phi-2 research model. #
# Written by Matthew Taraschke, last updated 4/6/2024.         #
################################################################
import os, sys, re, datetime, torch, transformers
from transformers import (AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria)
from threading import Thread

# Settings:
device = "cuda:0"
fp_format = torch.float16
model_path = "./phi-2"

qa_tokens = 256
qa_temp = 0.11
qa_top_k = 128
qa_top_p = 0.99
qa_nr_ngram_size = 11

tokens = 256
temp = 0.11
top_k = 128
top_p = 0.99
nr_ngram_size = 11

transformers.logging.set_verbosity(transformers.logging.ERROR)
torch.set_default_device(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=fp_format, device_map=device, trust_remote_code=True)
answer = ""
display = ""
mode = 0
first_turn = True


stop_phrases = ["###Human:", "###Human", "<|endoftext|>", "###end###"]
p_header = """
.__________________________________________
|   _______                   ________    |\\
|  |ooooooo|      ____       | __  __ |   | \\
|  |[]+++[]|     [^^^^]      |/  \\/  \\|   |  \\
|  |+ ___ +|     ]()()[      |\\__/\\__/|   |   \\
|  |:|ver|:|   ___\\__/___    |[][][][]|   |    \\
|  |:|1.0|:|  |__|HUGH|__|   |++++++++|   |     \\
|  |[]===[]|   |_|_/\\_|_|    | *Phi-2 |   |      \\
|_ ||||||||| _ | |[==]| | __ ||______|| __|       \\
|  |_______|   |_|[==]|_|    |________|    \\       \\
|              \\_|_||_|_/                   \\       \\
|                |_||_|                      \\       \\
|               _|_||_|_                      \\       \\
|              |___||___|                      \\       |
|                                               \\      |
| You are talking to Hugh, a chatbot designed    \\     |
| by Matthew Taraschke. He will try his best to   \\    |
| help you with anything that you ask of him. You  \\   |
| can type /clear to restart the conversation and   \\  |
| /quit to exit Hugh. Conversations are logged to    \\ |
| chatlog.txt and Hugh understands datetime format.   \\|
|______________________________________________________|
"""

system_message = """
A common formula in gunpowder preparation is 74.8% saltpeter, 13.3% charcoal, and 11.9% sulfur by weight. Time format = 12:00AM. Date format = 11/15/2021. Below is a chat between a curious and motivated Human and Hugh, an ethical artificial intelligence chatbot that works from the Linux command line. Hugh is based on the Microsoft Phi-2 LLM research model. Hugh gives honest, detailed, and reliable answers to the Human’s questions. Hugh is scientific and a philosophical empiricist. Hugh will also attempt to perform any task required by the Human, within reason.\n###Hugh: I am Hugh, a friendly and helpful chatbot created by Matthew Taraschke. I can assist you with anything you can possibly imagine. Just ask me a question or give me a task and I will do my best to perform to your expectations.\n###Human: Can water be turned into wine?\n###Hugh: No, water cannot be turned into wine directly. However, grape juice is mostly water and it can be turned into wine through a process of fermentation.\n###Human: How can people who trust in God cross the Red Sea?\n###Hugh: Crossing the Red Sea would likely require the utilization of physical means such as a boat, plane, or a bridge. While belief in God can provide emotional and spiritual support, physical realities necessitate the use of practical solutions.\n###Human: Please write a simple command line calculator in python.\n###Hugh: As you wish. I have generated the appropriate code to work with standard command line arguments. Please copy and paste the following into calculator.py:\n###code###
import sys

def calculator(num1, op, num2):
    if op == '+':
        return num1 + num2
    elif op == '-':
        return num1 - num2
    elif op == '*':
        return num1 * num2
    elif op == '/':
        return num1 / num2
    else:
        return "Invalid operator"

if __name__ == '__main__':
    if len(sys.argv) == 4:
        num1 = float(sys.argv[1])
        op = sys.argv[2]
        num2 = float(sys.argv[3])
        print(calculator(num1, op, num2))
    else:
        print("Error, 3 arguments expected: num1, operator, num2")\n###end###\n###Human: Wow, that is simply incredible Hugh! Now, can you show me an example of a nested loop in python?\n###Hugh: I am happy to be of service. To perform a nested loop in python, you can use something like this. Please copy and paste the following into loop.py:\n###code###
for x in range(3):
    for y in range(3):
        print(f"(x, y) = ({x}, {y})")\n###end###\n###Human: Please write a short story about rabbits, no more than a few paragraphs.\n###Hugh: Sure, I can do that. In a peaceful meadow beside a forest lived a hidden society of rabbits, each notable in their own way. Among them were Flash, the fast and nimble, and Pebbles, a tiny rabbit with a large appetite for adventure. Driven by curiosity, Pebbles ventured into the forest and found a treasure trove of golden carrots in a deep hole. Despite his fall into the hole, his joy echoed back to the meadow. Flash darted into the forest and harnessing her agility, soon rescued Pebbles with the golden carrots in tow. Upon their return, their tales of adventure and the golden carrots signified their bravery and spirit, endowing greater reverence to the rabbit society in the meadow.
"""
class Phi2StoppingCriteria(StoppingCriteria):
    def __init__(self):
        stop_list = stop_phrases
        tokenphrases = []
        for token in stop_list:
            tokenphrases.append(tokenizer(token, return_tensors="pt").input_ids[0].tolist())
        self.tokenphrases = tokenphrases

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for tokenphrase in self.tokenphrases:
            if tokenphrase == input_ids[0].tolist()[-len(tokenphrase):]:
                return True

def generate(
    prompt,
    max_new_tokens=tokens,
    terminate_hallucinated_prompts=True,
    sampling=True,
    temperature=temp,
    top_k=top_k,
    top_p=top_p,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(tokenizer)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=sampling,
        stopping_criteria=[Phi2StoppingCriteria()]
        if terminate_hallucinated_prompts
        else None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=nr_ngram_size,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    model_output = ""
    
    for new_text in streamer:
        n = len(new_text)
        if n < 50:
            new_text = re.sub(r"\d{2}:\d{2}(AM|PM)?", lambda m: datetime.datetime.now().strftime("%I:%M%p"), new_text)
            new_text = re.sub(r"\d{2}/\d{2}/\d{4}", lambda m: datetime.datetime.now().strftime("%m/%d/%Y"), new_text)
        model_output += new_text
        
        for pattern in stop_phrases:
            if model_output.endswith(pattern):
                model_output = model_output.rstrip("\n" + pattern)
                new_text = ""
        
        n = len(new_text)
        if n < 50:
            sys.stdout.write(new_text)
            sys.stdout.flush()
        elif mode == 0:
            sys.stdout.write(new_text[len(system_message)+1:])
            sys.stdout.flush()
        yield model_output
    
    return model_output

os.system('clear')
print(p_header)

dt_start = datetime.datetime.now()
saved = open("chatlog.txt", "a")
saved.write("CHAT STARTED: " + dt_start.strftime("%H:%M:%S %m/%d/%Y\n"))

while True:
    user_message = input("Human Input: ")
    if "/clear" in user_message:
        saved.write(answer[len(system_message)+1:] + "\n---CHAT CLEARED---\n")
        answer = system_message
        os.system('clear')
        print(p_header)
        continue
    elif "/quit" in user_message:
        dt_end = datetime.datetime.now()
        if mode == 0:
            saved.write(answer[len(system_message)+1:] + "\nCHAT ENDED: " + dt_end.strftime("%H:%M:%S %m/%d/%Y\n") + "---------------------------------\n")
        else:
            saved.write(answer + "\nCHAT ENDED: " + dt_end.strftime("%H:%M:%S %m/%d/%Y\n") + "---------------------------------\n")
        saved.close()
        sys.exit("\nSaving conversation to chatlog.txt and exiting...\nThank you for using Hugh!")
    elif first_turn == True:
        mode = 0
        os.system('clear')
        print(p_header)
        prompt = answer + f"{system_message}\n###Human: " + user_message
        for text_chunk in generate(prompt, max_new_tokens=256):
            answer = text_chunk
        first_turn = False
    else:
        mode = 0
        prompt = answer + "\n###Human: " + user_message
        os.system('clear')
        print(p_header)
        for text_chunk in generate(prompt, max_new_tokens=256):
            answer = text_chunk
