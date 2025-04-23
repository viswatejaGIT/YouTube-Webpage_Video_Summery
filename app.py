import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup

# Initialize LLM
def get_llm():
    return ChatGroq(
        groq_api_key="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",
        model_name="gemma2-9b-it"
    )

# YouTube Transcript Fetcher
def get_youtube_transcript(video_url):
    try:
        video_id = parse_qs(urlparse(video_url).query).get("v")
        if not video_id:
            video_id = video_url.split("/")[-1]  # fallback
        transcript = YouTubeTranscriptApi.get_transcript(video_id[0] if isinstance(video_id, list) else video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error: {str(e)}"

# Web Page Text Extractor
def get_webpage_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        return ' '.join(text.split())
    except Exception as e:
        return f"Error: {str(e)}"

# Run summarization
def summarize_text(text):
    llm = get_llm()
    prompt = PromptTemplate.from_template("Summarize the following content in 5 to 10 lines:\n\n{text}")
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    docs = [Document(page_content=text)]
    return chain.invoke(docs)["output_text"]

# ---------- Streamlit UI ----------
st.title("Summarize YouTube & Web URLs")

option = st.radio("Choose input type:", ["YouTube Video(your video should have english subtitles)", "Webpage URL(your url should be public)"])

url_input = st.text_input("Enter the URL here:")

if st.button("Summarize"):
    if not url_input:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Fetching and summarizing content..."):
            if option == "YouTube Video":
                raw_text = get_youtube_transcript(url_input)
            else:
                raw_text = get_webpage_text(url_input)

            if raw_text.startswith("Error"):
                st.error(raw_text)
            else:
                summary = summarize_text(raw_text)
                st.subheader("Summary:")
                st.write(summary)
