import streamlit as st
import pandas as pd
from inference import predict_text  # <-- import your custom prediction
from llama_explain import explain

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Streamlit Demo: Local Classifier")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "file" in msg and msg["file"]:
            st.markdown(msg["file"])

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
file_content = None
text_column = None

if uploaded_file:
    name = '.'.join(uploaded_file.name.split('.')[:-1])
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=None)

    if isinstance(df, dict):
        # Take first sheet
        first_sheet = list(df.keys())[0]
        df = df[first_sheet]

    # Detect text column automatically (first string column)
    text_cols = df.select_dtypes(include="object").columns
    if len(text_cols) > 0:
        text_column = text_cols[0]
    else:
        st.warning("No text column detected in file.")

    file_content = f"**{name} (showing first 10 rows of '{text_column}')**:\n\n" + df.head(10)[text_column].to_markdown()
    st.markdown(file_content)

# ----------------------------
# Chat input
# ----------------------------
if prompt := st.chat_input("Enter a message to classify:"):
    with st.chat_message("user"):
        st.markdown(prompt)
        if file_content:
            st.markdown(file_content)

    st.session_state.messages.append({"role": "user", "content": prompt, "file": file_content})

    # Classify user message
    pred_label = predict_text([prompt])[0]  # returns a list, take first element
    explanation = explain(prompt, pred_label)

    with st.chat_message("assistant"):
        st.markdown(f"**Predicted label:** {pred_label}")
        st.markdown(f"**Explanation:** {explanation}")
    
    st.session_state.messages.append({"role": "assistant", "content": f"Predicted label: {pred_label}\nExplanation: {explanation}"})

# ----------------------------
# Optional: classify uploaded file column
# ----------------------------
if uploaded_file and text_column:
    st.markdown("---")
    st.markdown("### Classifying uploaded file")

    df_sample = df.head(10)
    texts = df_sample[text_column].tolist()
    predictions = predict_text(texts)

    results_df = pd.DataFrame({
        text_column: df_sample[text_column],
        "Predicted Label": predictions
    })

    st.dataframe(results_df)