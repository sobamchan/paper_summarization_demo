import streamlit as st
from schnitsum import SchnitSum


st.title("SCITLDR demo.")

txt = st.text_area(
    "Text to analyze",
    """Modern machine learning models are opaque, and as a result there is a burgeoning academic subﬁeld on methods that explain these models’ behavior. However, what is the precise goal of providing such explanations, and how can we demonstrate that explanations achieve this goal? Some research argues that explanations should help teach a student (either human or machine) to simulate the model being explained, and that the quality of explanations can be measured by the simulation accuracy of students on unexplained examples. In this work, leveraging meta-learning techniques, we extend this idea to improve the quality of the explanations themselves, speciﬁcally by optimizing explanations such that student models more effectively learn to simulate the original model. We train models on three natural language processing and computer vision tasks, and ﬁnd that students trained with explanations extracted with our framework are able to simulate the teacher signiﬁcantly more effectively than ones produced with previous methods. Through human annotations and a user study, we further ﬁnd that these learned explanations more closely align with how humans would explain the required decisions in these tasks. Our code is available at https://github.com/coderpat/learning-scaffold.
     """,
)

model = SchnitSum("sobamchan/bart-large-scitldr")
st.subheader("TLDR: ")
st.write(model([txt])[0])
