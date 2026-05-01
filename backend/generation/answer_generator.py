"""
Generate cited answers using Gemini or Groq.
"""
import httpx
from backend.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    GEMINI_MODEL,
    GROQ_MODEL,
)


SYSTEM_PROMPT = (
    "You are an academic research assistant. Based on the provided research paper abstracts, "
    "answer the question concisely and accurately. Use only information from the provided abstracts. "
    "Cite papers using their number markers like [1], [2], etc. "
    "If the abstracts don't contain enough information to fully answer, say so clearly."
)


def _build_prompt(query: str, top_papers: list[dict]) -> str:
    context = ""
    for i, paper in enumerate(top_papers, 1):
        context += f"[{i}] {paper['title']}\n{paper['abstract']}\n\n"

    return (
        f"Papers:\n{context}\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


async def generate_answer_gemini(query: str, top_papers: list[dict]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt(query, top_papers)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


async def generate_answer_groq(query: str, top_papers: list[dict]) -> str:
    from groq import AsyncGroq

    client = AsyncGroq(api_key=GROQ_API_KEY)
    prompt = _build_prompt(query, top_papers)
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


async def generate_answer(query: str, top_papers: list[dict]) -> str:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "groq":
        return await generate_answer_groq(query, top_papers)
    return await generate_answer_gemini(query, top_papers)
