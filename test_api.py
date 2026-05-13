import google.generativeai as genai
import sys

try:
    genai.configure(api_key="REDACTED")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("hello")
    print("SUCCESS:", response.text)
except Exception as e:
    print("ERROR:", str(e))
