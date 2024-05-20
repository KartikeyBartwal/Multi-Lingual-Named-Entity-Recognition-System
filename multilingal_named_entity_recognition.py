import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("asahi417/tner-xlm-roberta-large-panx-dataset-en")
model = AutoModelForTokenClassification.from_pretrained("asahi417/tner-xlm-roberta-large-panx-dataset-en")

print(model)

# Stunning Background Image
page_bg_img = '''
<style>
body {
  background-image: url("https://images.unsplash.com/photo-1584283520432-39545c39e43a?fit=crop&w=1920&h=1080");
  background-size: cover;
  background-attachment: fixed;
  font-family: sans-serif;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Informative Title
st.title("Multilingual Named Entity Recognition")

st.write(model)
# Interactive Text Input with Stylish Design
text = st.text_input("Enter the text...", key="user_text")

# Button with Improved Design
analyze_button = st.button("Analyze")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)


# Output with Dynamic Text Color
if analyze_button:
    # Perform named entity recognition on the input text
    entities = nlp(text)
    print(entities)

    st.write(entities)

    # Display the results
    st.write("Detected Entities:")
    for entity in entities:
        print(entity)
