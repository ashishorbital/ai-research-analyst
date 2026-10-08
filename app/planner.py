import json
import os

from dotenv import load_dotenv
from groq import Groq

from .models import ResearchPlan


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def create_research_plan(query: str) -> ResearchPlan:

    prompt = f"""
You are an expert research analyst.

Break the following research question into 4-6
independent research questions.

The questions should cover:
- important facts
- quantitative evidence
- opposing viewpoints
- risks
- relevant context

Research question:

{query}

Return ONLY valid JSON in this format:

{{
  "original_query": "...",
  "questions": [
    {{
      "question": "...",
      "purpose": "..."
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content

    data = json.loads(content)

    return ResearchPlan(**data)