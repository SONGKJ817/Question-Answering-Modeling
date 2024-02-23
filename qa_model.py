import curses
import re
from transformers import ElectraForQuestionAnswering, ElectraTokenizerFast
from transformers import DebertaV2ForQuestionAnswering, DebertaV2TokenizerFast
from transformers import RobertaForQuestionAnswering, RobertaTokenizerFast
from transformers import pipeline
import re
import torch
import os

## Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Path 
base_path = os.getcwd()
data_path = os.path.join(base_path,'data')
model_path = os.path.join(base_path,'model')

model_dir1 = os.path.join(model_path,'koelectra')
model_dir2 = os.path.join(model_path,"mdeberta")
model_dir3 = os.path.join(model_path, 'xlm_roberta')
## Model upload

model1 = ElectraForQuestionAnswering.from_pretrained(model_dir1)
tokenizer1 = ElectraTokenizerFast.from_pretrained(model_dir1)

model2 = DebertaV2ForQuestionAnswering.from_pretrained(model_dir2)
tokenizer2 = DebertaV2TokenizerFast.from_pretrained(model_dir2)

model3 = RobertaForQuestionAnswering.from_pretrained(model_dir3)
tokenizer3 = RobertaTokenizerFast.from_pretrained(model_dir3)

nlp1 = pipeline('question-answering', model=model1, tokenizer = tokenizer1, device=device)
nlp2 = pipeline('question-answering', model=model2, tokenizer = tokenizer2, device=device)
nlp3 = pipeline('question-answering', model=model3, tokenizer = tokenizer3, device=device)

models = [nlp1, nlp2, nlp3]
context = ""

def file_upload(title):
    global context
    file_path = os.path.join(data_path, f"{title}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = re.sub(r'•|\n', '', text)
    context = text


def process_text(question):
    context_score_rank = len(context) * [0]
    context_freq_rank = len(context) * [0]

    start_ids = []
    end_ids = []
    score_ids = []

    for model in models:
        result = model({'question': question, 'context': context})

    start_ids.append(result["start"])
    end_ids.append(result["end"])
    score_ids.append(result["score"])

    for i, j, k in zip(start_ids, end_ids, score_ids):
        context_score_rank[i:j+1] = [x+k for x in context_score_rank[i:j+1]]
        context_freq_rank[i:j+1] = [x+1 for x in context_freq_rank[i:j+1]]

    max_score_value = max(context_score_rank)
    max_freq_value = max(context_freq_rank)

    max_score_indices = [i for i, value in enumerate(context_score_rank) if value > max_score_value*0.8]
    max_freq_indices = [i for i, value in enumerate(context_freq_rank) if value > max_freq_value*0.8]

    result_score = [context[x] for x in max_score_indices]
    result_score = ''.join(result_score)
    result_freq = [context[x] for x in max_freq_indices]
    result_freq = ''.join(result_freq)

    return result_score, result_freq

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor
    stdscr.clear()

    import locale

    # UTF-8 지원 활성화
    locale.setlocale(locale.LC_ALL, '')

    # Set up color pairs
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    highlight_color = curses.color_pair(1)

    choices = ["롯데칠성", "자동차", "퀄컴", "하이브", "미국 마감시황"]
    choice_idx = 0

    while True:
        stdscr.clear()

        # Display menu options
        for i, option in enumerate(choices):
            if i == choice_idx:
                stdscr.addstr(i + 1, 1, option, highlight_color)
            else:
                stdscr.addstr(i + 1, 1, option)

        stdscr.refresh()

        # Get user input
        key = stdscr.getch()

        if key == curses.KEY_UP and choice_idx > 0:
            choice_idx -= 1
        elif key == curses.KEY_DOWN and choice_idx < len(choices) - 1:
            choice_idx += 1
        elif key == 10:  # Enter key
            if choice_idx == len(choices) - 1:
                break
            else:
                title = choices[choice_idx]
                file_upload(title)
                stdscr.addstr(len(choices) + 2, 1, f"Selected: {title}", highlight_color)
                stdscr.refresh()

                stdscr.addstr(len(choices) + 3, 1, "Enter your question:")
                stdscr.refresh()

                curses.echo()  # Enable echoing of input
                
                question = stdscr.getstr(len(choices) + 4, 1).decode('utf-8')
                print(question)

                curses.noecho()  # Disable echoing of input

                result_score, result_freq = process_text(question)

                stdscr.addstr(f"Answer(score): {result_score}")
                stdscr.refresh()

                stdscr.getch()  # Wait for user to press a key

if __name__ == "__main__":
    curses.wrapper(main)
