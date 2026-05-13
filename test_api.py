import os

import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("hello")
    print("SUCCESS:", response.text)
except Exception as e:
    print("ERROR:", str(e))
