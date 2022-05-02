import streamlit as st
from transformers import (AutoTokenizer, BartConfig,
                          BartForConditionalGeneration)
import torch


class Model:
    def __init__(self, model_path):
        print("initializing...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        bart = BartForConditionalGeneration(BartConfig())
        bart.load_state_dict(
            torch.load(model_path),
            strict=False,
        )
        self.bart = bart
        print("loaded!")

    def summarize(self, text: str):

        inputs = self.tokenizer(
            [text], padding="max_length", truncation=True, return_tensors="pt"
        )
        summary_ids = self.bart.generate(
            inputs["input_ids"],
            max_length=50,
            num_beams=1,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]


st.title("SCITLDR demo.")

txt = st.text_area(
    "Text to analyze",
    """Modern machine learning models are opaque, and as a result there is a burgeoning academic subﬁeld on methods that explain these models’ behavior. However, what is the precise goal of providing such explanations, and how can we demonstrate that explanations achieve this goal? Some research argues that explanations should help teach a student (either human or machine) to simulate the model being explained, and that the quality of explanations can be measured by the simulation accuracy of students on unexplained examples. In this work, leveraging meta-learning techniques, we extend this idea to improve the quality of the explanations themselves, speciﬁcally by optimizing explanations such that student models more effectively learn to simulate the original model. We train models on three natural language processing and computer vision tasks, and ﬁnd that students trained with explanations extracted with our framework are able to simulate the teacher signiﬁcantly more effectively than ones produced with previous methods. Through human annotations and a user study, we further ﬁnd that these learned explanations more closely align with how humans would explain the required decisions in these tasks. Our code is available at https://github.com/coderpat/learning-scaffold.
     """,
)

model = Model("./model.state")
st.subheader("TLDR: ")
st.write(model.summarize(txt))
