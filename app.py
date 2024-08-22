import numpy as np
import re
import json
import streamlit as st
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAI

# Load Google API key from secrets
google_api_key = st.secrets['GOOGLE_AI']['google_api_key']
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

st.header("ATS Insight")
file_upload = st.file_uploader("Upload a file", type=["pdf"])

def get_key_dict(words):
    w = re.findall(r'\{[^{}]*\}', words)
    key_data = json.loads(w[0])
    arr = list(key_data.values())
    n1 = np.array(arr, dtype=object)
    flat_array = np.concatenate(arr).flatten()
    final_set = []
    for i in flat_array:
        if len(i) > 0:
            arr = i.split(' ')
            for k in arr:
                if len(k) > 0 and not any(chr.isdigit() for chr in k):
                    final_set.append(k.lower().replace('(', '').replace(')', ''))
    no_count = [
        "about", "above", "across", "after", "against",
        "along", "amid", "among", "around", "at",
        "before", "behind", "below", "beneath", "beside",
        "between", "beyond", "by", "down", "during",
        "for", "from", "in", "inside", "into",
        "near", "of", "off", "on", "out",
        "outside", "over", "past", "through", "to",
        "toward", "under", "underneath", "until", "up",
        "upon", "with", "within", "the", "an", "a", "and", "but", "or", "nor", "yet", "so",
        "although", "because", "since",
        "unless", "until", "while"
    ]
    final_set = list(set(final_set))
    
    compare_dict = {}
    for i in final_set:
        if i not in no_count:
            compare_dict[i] = compare_dict.get(i, 0) + 1
    return compare_dict

def get_job_keywords(job, level_option):
    if level_option and job:
        prompt = f"Give keywords for the resume for an {level_option} in {job} role in three sections: 'Work experience', 'Projects', 'Skills'. The response should be in JSON format."
        words = llm.invoke(prompt)
        compare_dict = get_key_dict(words)
        return compare_dict

if file_upload is not None:
    pdf_reader = PdfReader(file_upload)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    if text:
        text_split = text.split('\n')
        user_words = []
        for line in text_split:
            arr = line.split(' ')
            for word in arr:
                if word:
                    user_words.append(word.lower())
        
        if user_words:
            user_words = list(set(user_words))
            job = st.text_input("Enter the Job Role:")
            level_option = st.selectbox("Select the level of your role:", ("Internship", "Entry Level"), index=None, placeholder="Select the Job Level...")
            but = st.button("Submit")
            
            if but:
                with st.spinner("Scanning, please wait..."):
                    ATS_score = 0
                    all_missing_words = ''
                    missing_words = []
                    
                    for _ in range(3):
                        compare_dict = get_job_keywords(job, level_option)
                        for word in user_words:
                            if word in compare_dict:
                                compare_dict[word] = compare_dict.get(word) + 1
                        
                        user_points = 0
                        for key in compare_dict:
                            if compare_dict[key] > 1:
                                user_points += 1
                            else:
                                missing_words.append(key)
                        
                        total_points = len(compare_dict)
                        user_score = (user_points / total_points) * 100
                        ATS_score += user_score
                    
                    net_score = ATS_score / 3
                    missing_words = list(set(missing_words))
                    
                    st.markdown("# ATS Score (based on keyword analysis)")
                    st.markdown(f"## {net_score:.2f}/100")
                    
                    all_missing_words = ', '.join(missing_words)
                    st.markdown("### Suggestions for Keywords")
                    st.code(all_missing_words, language='markdown')
