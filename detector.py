"""
detector.py — Backend pipeline for Explainable Fake News Detection

This module implements a Search-Augmented Generation (SAG) pipeline:
- RETRIEVAL: search_related_sources() fetches live web articles via Tavily
- AUGMENTATION: format_sources() formats retrieved articles for the LLM prompt
- GENERATION: Groq LLM generates a structured verdict with full reasoning
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _get_api_key(key_name: str) -> str:
    """
    Gets an API key from environment variables or Streamlit secrets.

    Checks in order:
    1. os.environ (set by .env via load_dotenv)
    2. st.secrets (for Streamlit Cloud deployment)

    Args:
        key_name: The name of the API key (e.g., "GROQ_API_KEY")

    Returns:
        str: The API key value

    Raises:
        ValueError: If the key is not found in either location
    """
    # First try os.environ (local .env file)
    key = os.environ.get(key_name)

    if key:
        return key

    # Fallback to Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass

    raise ValueError(
        f"{key_name} not found. Set it in .env file locally or in "
        "Streamlit Cloud secrets for deployment."
    )


def initialize_tools() -> tuple:
    """
    Initializes and returns the LLM and search tool.

    Loads API keys from .env file (local) or st.secrets (Streamlit Cloud).
    Instantiates ChatGroq with llama-3.3-70b-versatile model at temperature 0.1.
    Instantiates TavilySearch with max_results=5.

    Returns:
        tuple: (llm, search_tool) where llm is a ChatGroq instance
               and search_tool is a TavilySearch instance
    """
    # Load .env file (works locally, ignored on Streamlit Cloud)
    load_dotenv()

    # Ensure API keys are available (fallback to st.secrets for Streamlit Cloud)
    # This also sets them in os.environ if they came from st.secrets
    groq_key = _get_api_key("GROQ_API_KEY")
    tavily_key = _get_api_key("TAVILY_API_KEY")

    # Set environment variables explicitly (needed for Streamlit Cloud)
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    # Get API keys with fallback to st.secrets
    groq_key = _get_api_key("GROQ_API_KEY")
    tavily_key = _get_api_key("TAVILY_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=groq_key
    )

    search_tool = TavilySearch(
        max_results=5,
        tavily_api_key=tavily_key
    )

    return llm, search_tool


def search_related_sources(claim: str, search_tool) -> list:
    """
    Runs two Tavily queries, merges results, and de-duplicates by URL.

    Query 1: The raw claim — catches news articles reporting the same story
    Query 2: "fact check: {claim}" — surfaces fact-checking sites like Snopes, PolitiFact

    Args:
        claim: The news claim or headline to search for
        search_tool: TavilySearch instance

    Returns:
        list: Max 6 unique search results, each containing title, url, content
    """
    # Query 1: Raw claim
    response_raw = search_tool.invoke(claim)

    # Query 2: Fact-check prefixed query
    response_factcheck = search_tool.invoke(f"fact check: {claim}")

    # TavilySearch returns a dict with 'results' key containing the list
    # Extract the results list from the response
    results_raw = response_raw.get("results", []) if isinstance(response_raw, dict) else []
    results_factcheck = response_factcheck.get("results", []) if isinstance(response_factcheck, dict) else []

    # Merge results
    all_results = []
    all_results.extend(results_raw)
    all_results.extend(results_factcheck)

    # De-duplicate by URL (keep first occurrence)
    seen_urls = set()
    unique_results = []
    for result in all_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Return max 6 results
    return unique_results[:6]


def format_sources(search_results: list) -> str:
    """
    Formats search results as numbered SOURCE blocks for the LLM prompt.

    Args:
        search_results: List of search result dicts with title, url, content fields

    Returns:
        str: Formatted string with numbered SOURCE blocks
    """
    if not search_results:
        return "No sources available."

    formatted = []
    for i, result in enumerate(search_results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        content = result.get("content", "No content available")

        block = f"""SOURCE {i}:
Title: {title}
URL: {url}
Content: {content}"""
        formatted.append(block)

    return "\n\n".join(formatted)


def create_analysis_prompt() -> ChatPromptTemplate:
    """
    Creates and returns the LangChain ChatPromptTemplate for fact-checking.

    The prompt instructs the LLM to:
    - Act as an expert fact-checker
    - Base verdict STRICTLY on provided sources, not prior training knowledge
    - Cite sources inline by number using (Source N) format
    - Output in exactly 6 sections: VERDICT, CONFIDENCE, EVIDENCE SUMMARY,
      REASONING, RED FLAGS, SOURCES CONSULTED

    Returns:
        ChatPromptTemplate: Template with system and human messages
    """
    system_prompt = """You are an expert fact-checker with years of experience verifying news claims. Your task is to analyze the given claim using ONLY the provided sources and deliver a structured verdict.

CRITICAL RULES:
1. Base your verdict STRICTLY on the provided sources. Do NOT use your training knowledge to judge the claim.
2. If the sources clearly contradict the claim, mark the verdict as FALSE.
3. If the sources do not provide enough information to make a definitive judgment, mark the verdict as UNVERIFIED.
4. Always cite sources inline by number using (Source N) format in your reasoning.
5. Be objective and neutral in your analysis.

You must respond in EXACTLY this format with these 6 sections:

VERDICT: [TRUE / FALSE / MISLEADING / UNVERIFIED]

CONFIDENCE: [HIGH / MEDIUM / LOW]

EVIDENCE SUMMARY:
- [Key finding from sources]
- [Another key finding]
- [Additional relevant findings]

REASONING:
[Write 2-3 paragraphs explaining your verdict. Cite specific sources inline using (Source 1), (Source 2), etc. Explain what the sources say vs what the claim says. Highlight any discrepancies or confirmations.]

RED FLAGS:
[List specific misinformation patterns detected such as: sensationalist language, lack of credible sources, logical fallacies, doctored images, out-of-context quotes, etc. Write "None detected" if no red flags are found.]

SOURCES CONSULTED:
[List the URLs of the sources most relevant to your verdict, one per line]"""

    human_prompt = """CLAIM TO VERIFY:
{claim}

AVAILABLE SOURCES:
{sources}

Based on the sources above, provide your fact-check analysis in the required format."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    return prompt


def analyze_claim(claim: str) -> dict:
    """
    Orchestrates the full fact-checking pipeline.

    Pipeline:
    1. Initialize LLM and search tools
    2. Search for related sources (two queries, merged, de-duplicated)
    3. Format sources for the prompt
    4. Build and run the LangChain chain
    5. Return structured result

    Args:
        claim: The news claim or headline to fact-check

    Returns:
        dict: {
            "claim": str,           # The original claim
            "analysis": str,        # Full LLM analysis output
            "raw_sources": list,    # Raw search results from Tavily
            "num_sources": int      # Number of sources used
        }
    """
    # Initialize tools
    llm, search_tool = initialize_tools()

    # Search for related sources
    search_results = search_related_sources(claim, search_tool)

    # Handle case where no sources found
    if not search_results:
        return {
            "claim": claim,
            "analysis": "No sources found. Unable to verify the claim due to lack of available information. Please try a different search term or check if the claim is too recent or obscure.",
            "raw_sources": [],
            "num_sources": 0
        }

    # Format sources
    formatted_sources = format_sources(search_results)

    # Create prompt
    prompt = create_analysis_prompt()

    # Build and run chain
    chain = prompt | llm | StrOutputParser()

    analysis = chain.invoke({
        "claim": claim,
        "sources": formatted_sources
    })

    return {
        "claim": claim,
        "analysis": analysis,
        "raw_sources": search_results,
        "num_sources": len(search_results)
    }


# For testing the module directly
if __name__ == "__main__":
    test_claim = "The Great Wall of China is visible from space"
    print(f"Testing with claim: {test_claim}\n")
    result = analyze_claim(test_claim)
    print("=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print(result["analysis"])
    print("=" * 60)
    print(f"Sources used: {result['num_sources']}")
