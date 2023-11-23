from transformers import pipeline
import pandas as pd

if __name__ == '__main__':
    text = '''It started before I was born. My biological mother was a young, unwed college graduate student, and she decided to put me up for adoption. She felt very strongly that I should be adopted by college graduates, so everything was all set for me to be adopted at birth by a lawyer and his wife. Except that when I popped out they decided at the last minute that they really wanted a girl. So my parents, who were on a waiting list, got a call in the middle of the night asking: "We have an unexpected baby boy; do you want him?" They said: "Of course." My biological mother later found out that my mother had never graduated from college and that my father had never graduated from high school. She refused to sign the final adoption papers. She only relented a few months later when my parents promised that I would someday go to college.'''

    # 1. 감성 분석
    # classifier = pipeline("text-classification")
    # output = classifier(text)
    # result = pd.DataFrame(output)

    # 2. 개체명 인식
    # ner_tagger = pipeline('ner', aggregation_strategy='simple')
    # output = ner_tagger(text)
    # result = pd.DataFrame(output)


    # 3. QnA
    # reader = pipeline("question-answering")
    # question= "What is his mother job?"
    # output = reader(question=question, context=text)
    # print(output)

    # 4. Summary
    summarizer = pipeline("summarization")
    output = summarizer(text, max_length=100, clean_up_tokenization_spaces=True)
    print(output[0]['summary_text'])