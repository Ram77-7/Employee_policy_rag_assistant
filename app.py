import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/response"

st.title("Employee Policy RAG Assistant")

question = st.text_input("Enter your question", placeholder="Ask something ....")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    API_URL,
                    json={"query": question},
                    timeout=60,
                )
            response.raise_for_status()
            st.markdown("🤖 AI Generated Answer")
            st.markdown(f""" 
            ### 💡 Answer 
             {response.json()}
            
            ---- 
            *Sources: ** Employee leave policy
            
            *Status: ** Relevant Information Found
            """)  # or response.json()["answer"] if it returns a dict
            if response.status_code == 200:
                st.success("AI response generated successfully!")
            else:
                st.error("Got an error with rag Assistance. Please try after some time!")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not get a response from the API: {e}")
