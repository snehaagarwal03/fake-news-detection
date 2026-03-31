# Explainable Fake News Detection using LLMs with Multi-Source Reasoning

---

## What Is This Project About?

Every day, millions of news articles are shared online. Many are genuine, but a significant portion is misleading, fabricated, or heavily biased. Most people cannot easily tell the difference, and existing automated fact-checking tools usually just give a "real" or "fake" label without explaining *why* — which makes them hard to trust.

**This project solves that problem.**

We are building a system where you can paste any news headline or article text, and the system will:

1. **Search the live internet** for related reporting from multiple sources (news outlets, fact-checking sites, etc.).
2. **Cross-reference** what the claim says against what multiple sources report.
3. **Produce a verdict** — TRUE, FALSE, MISLEADING, or UNVERIFIED — with a confidence level.
4. **Write a clear, human-readable explanation** of *why* it reached that verdict, with inline source citations.

The key differentiator is **explainability**. The system does not just say "this is fake." It tells you: "The article claims X, but according to Reuters and Snopes (Source 2, Source 4), the actual data shows Y. Based on this discrepancy, the claim is MISLEADING."

Think of it as an AI-powered research assistant that does the legwork of fact-checking for you and presents its findings with full source citations so you can verify everything yourself.

---

## Core Architecture — Search-Augmented Generation (SAG)

This system implements a **Search-Augmented Generation** pipeline — a modern, web-based variant of RAG where retrieval happens from the live internet instead of a local vector database. There is no ChromaDB, no sentence-transformers, no embeddings, no local vector store, and no dataset loading pipeline anywhere in this project. These are not needed because Tavily retrieves fresh, relevant sources on demand at runtime.

The three RAG stages map to the codebase as follows:

- **RETRIEVAL** → `search_related_sources()` fetches live web articles via Tavily
- **AUGMENTATION** → `format_sources()` formats retrieved articles and injects them into the LLM prompt as grounded context
- **GENERATION** → Groq LLM reads the injected sources and generates a structured verdict with full reasoning

No dataset (LIAR, FakeNewsNet, or any other) is part of the runtime pipeline. Datasets are only referenced during manual evaluation after the app is built — by copying headlines from Snopes.com into the running app and checking verdict accuracy. No dataset needs to be downloaded or loaded in code.

### Architecture Diagram

```text
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│              (user pastes claim text)                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│                 detector.py                          │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  RETRIEVAL                                   │   │
│  │  search_related_sources(claim)               │   │
│  │  ├─ Tavily query: "{claim}"                  │   │
│  │  ├─ Tavily query: "fact check: {claim}"      │   │
│  │  ├─ Merge results, de-duplicate by URL       │   │
│  │  └─ Return max 6 unique sources              │   │
│  └────────────────┬─────────────────────────────┘   │
│                   │                                  │
│                   ▼                                  │
│  ┌──────────────────────────────────────────────┐   │
│  │  AUGMENTATION                                │   │
│  │  format_sources(results)                     │   │
│  │  └─ Format each source as numbered block     │   │
│  │     SOURCE 1: Title | URL | Content          │   │
│  │     SOURCE 2: Title | URL | Content          │   │
│  │     ...                                      │   │
│  └────────────────┬─────────────────────────────┘   │
│                   │                                  │
│                   ▼                                  │
│  ┌──────────────────────────────────────────────┐   │
│  │  GENERATION                                  │   │
│  │  Groq LLM (llama-3.3-70b-versatile, t=0.1)  │   │
│  │  Prompt = system + claim + formatted sources  │   │
│  │  Returns structured output:                  │   │
│  │    VERDICT / CONFIDENCE / EVIDENCE SUMMARY   │   │
│  │    REASONING / RED FLAGS / SOURCES CONSULTED │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Streamlit UI displays                   │
│  ├─ Color-coded verdict badge                       │
│  ├─ Full analysis report (left column, 2/3 width)  │
│  ├─ Source cards as expanders (right column, 1/3)   │
│  └─ Download .txt report button                     │
└─────────────────────────────────────────────────────┘
```

---

## What the System Must Deliver (Non-Negotiable)

1. Accept any news headline or article text as input
2. Execute two separate Tavily search queries per claim — one raw claim query and one prefixed with `"fact check: {claim}"` — to maximise source diversity and specifically surface fact-checking websites like Snopes, PolitiFact, AFP Fact Check
3. Merge both result sets, de-duplicate by URL, and pass up to 6 sources to the LLM
4. The LLM must return a response with exactly these labeled sections:
   - `VERDICT:` — one of: TRUE / FALSE / MISLEADING / UNVERIFIED
   - `CONFIDENCE:` — one of: HIGH / MEDIUM / LOW
   - `EVIDENCE SUMMARY:` — bullet points of key findings from sources
   - `REASONING:` — 2-3 paragraphs with inline source citations by number e.g. (Source 1), (Source 3)
   - `RED FLAGS:` — specific misinformation patterns detected, or "None detected"
   - `SOURCES CONSULTED:` — URLs most relevant to the verdict
5. The LLM prompt must explicitly instruct the model to base its verdict strictly on retrieved evidence, not on prior training knowledge, and to cite sources inline by number
6. The UI must parse the VERDICT field and render it color-coded (green=TRUE, red=FALSE, orange=MISLEADING, yellow=UNVERIFIED), display all retrieved sources as expandable cards, and offer a downloadable `.txt` report

---

## Tech Stack (Locked, No Deviations)

| Component | Tool | Why | Cost |
| --- | --- | --- | --- |
| Package manager | `uv` | 10-100x faster than pip, manages Python + venvs + deps in one tool | Free |
| Python | 3.11 | Stable, well-supported, installed and managed via uv | Free |
| LLM | Groq API, model `llama-3.3-70b-versatile`, temperature 0.1 | Free tier, extremely fast inference, strong reasoning model | Free (30 req/min) |
| Web Search | Tavily API (free tier) | Purpose-built for AI agents, clean structured results | Free (1000 searches/month) |
| Orchestration | LangChain (`langchain`, `langchain-groq`, `langchain-community`) | Chain composition with pipe syntax `prompt \| llm \| parser` | Free (open source) |
| Frontend | Streamlit | Simple Python web UI, wide layout, color-coded output | Free (open source) |
| Env management | `python-dotenv` | Loads API keys from `.env`, never hardcoded | Free |
| Deployment | Streamlit Community Cloud | Free hosting, connect GitHub repo | Free |

All commands are prefixed with `uv run`. No pip, no requirements.txt, no manual venv activation.

---

## Final File Structure (Exactly This, Nothing More)

```text
fake-news-detection/
├── .env                  # NEVER pushed — real API keys
├── .env.example          # Pushed — placeholder keys
├── .gitignore            # Pushed — excludes .env and .venv/
├── .python-version       # Pushed — set to 3.11
├── pyproject.toml        # Pushed — auto-generated by uv
├── uv.lock               # Pushed — exact dependency versions
├── detector.py           # Pushed — backend pipeline (all 5 functions)
├── app.py                # Pushed — Streamlit frontend
├── project.md            # Pushed — this file
└── README.md             # Pushed — documentation (Phase 5)
```

---

## Roadmap — Phase by Phase

---

### PHASE 1: ENVIRONMENT SETUP

---

#### Step 1: Install uv

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your terminal (or run: source ~/.zshrc)
# Then verify:
uv --version
```

#### Step 2: Install Python 3.11 via uv

```bash
# uv handles Python installation — no need for python.org or homebrew
uv python install 3.11

# Verify:
uv python list
```

#### Step 3: Initialize the Project

```bash
# Navigate to the project folder
cd /Users/snehaagarwal/Projects/fake-news-detection

# Initialize a uv project (creates pyproject.toml, .python-version, hello.py)
uv init

# Remove the starter file (we don't need it)
rm hello.py
```

#### Step 4: Install All Dependencies in One Command

```bash
uv add streamlit langchain langchain-groq langchain-tavily python-dotenv
```

What each package does:

- `streamlit` — Web UI framework
- `langchain` — LLM chain composition and prompt templates
- `langchain-groq` — Groq API integration for LangChain
- `langchain-tavily` — LangChain integration for Tavily search (replaces deprecated langchain-community Tavily tool)
- `python-dotenv` — Loads API keys from `.env` file

#### Step 5: Get API Keys (All Free)

**Groq (LLM API):**

1. Go to <https://console.groq.com>
2. Sign up with Google or GitHub (no credit card)
3. Go to "API Keys" in the left sidebar
4. Click "Create API Key"
5. Copy the key
6. Free tier: 30 requests/minute, ~14,400 requests/day

**Tavily (Web Search):**

1. Go to <https://tavily.com>
2. Sign up (no credit card)
3. Go to your dashboard to find your API key
4. Free tier: 1,000 searches/month

#### Step 6: Create `.env`, `.env.example`, and `.gitignore`

```bash
# .env — real keys, NEVER push this
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
EOF
```

Replace `your_groq_key_here` and `your_tavily_key_here` with your actual keys.

```bash
# .env.example — placeholder, this IS pushed to GitHub
cat > .env.example << 'EOF'
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly-your_key_here
EOF
```

```bash
# .gitignore
cat > .gitignore << 'EOF'
.env
.venv/
__pycache__/
*.pyc
.DS_Store
EOF
```

**Phase 1 done.** You now have a working Python environment with all dependencies installed and API keys configured.

---

### PHASE 2: BACKEND — `detector.py`

This file contains the entire backend pipeline. Exactly 5 functions, implemented in this order.

---

#### Step 7: `initialize_tools()`

**Signature:**

```python
def initialize_tools() -> tuple:
    """Returns (llm, search_tool) after loading API keys from .env."""
```

**What it does:**

1. Calls `load_dotenv()` to read `.env`
2. Instantiates `ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)`
3. Instantiates `TavilySearch(max_results=5)`
4. Returns `(llm, search_tool)` as a tuple

**Key imports needed:**

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
```

---

#### Step 8: `search_related_sources(claim, search_tool)`

**Signature:**

```python
def search_related_sources(claim: str, search_tool) -> list:
    """Runs two Tavily queries, merges and de-duplicates, returns max 6 results."""
```

**What it does:**

1. Runs `search_tool.invoke(claim)` — raw claim query
2. Runs `search_tool.invoke(f"fact check: {claim}")` — fact-check prefixed query
3. Merges both result lists
4. De-duplicates by URL field (keeps first occurrence)
5. Returns a list of max 6 unique results

**Why two queries:** The raw query catches news articles reporting the same story. The `"fact check:"` prefixed query specifically surfaces fact-checking sites like Snopes, PolitiFact, and AFP Fact Check. Together they give the LLM diverse, high-quality evidence.

---

#### Step 9: `format_sources(search_results)`

**Signature:**

```python
def format_sources(search_results: list) -> str:
    """Formats each result as a numbered SOURCE block. Returns a single string."""
```

**What it does:**

Takes the list of search results and formats each one as:

```text
SOURCE 1:
Title: <title>
URL: <url>
Content: <content snippet>

SOURCE 2:
Title: <title>
URL: <url>
Content: <content snippet>
...
```

Returns the entire thing as a single formatted string. This string gets injected into the LLM prompt as the `{sources}` template variable.

---

#### Step 10: `create_analysis_prompt()`

**Signature:**

```python
def create_analysis_prompt() -> ChatPromptTemplate:
    """Returns the LangChain ChatPromptTemplate with system + human messages."""
```

**What it does:**

Returns a `ChatPromptTemplate` with two messages:

**System message** — A detailed prompt that:

- Identifies the LLM as an expert fact-checker
- Instructs it to base verdict **strictly on provided sources**, not on prior training knowledge
- Instructs it to cite sources inline by number using `(Source N)` format
- Requires output in exactly this 6-section format:

```text
VERDICT: [TRUE / FALSE / MISLEADING / UNVERIFIED]

CONFIDENCE: [HIGH / MEDIUM / LOW]

EVIDENCE SUMMARY:
- [bullet point findings from sources]

REASONING:
[2-3 paragraphs with inline source citations like (Source 1), (Source 3)]

RED FLAGS:
[specific misinformation patterns detected, or "None detected"]

SOURCES CONSULTED:
[URLs most relevant to the verdict]
```

**Human message** — Template with two variables: `{claim}` and `{sources}`

**Key imports needed:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

---

#### Step 11: `analyze_claim(claim: str) -> dict`

**Signature:**

```python
def analyze_claim(claim: str) -> dict:
    """Orchestrates the full pipeline. Returns dict with keys: claim, analysis, raw_sources, num_sources."""
```

**What it does:**

1. Calls `initialize_tools()` to get `(llm, search_tool)`
2. Calls `search_related_sources(claim, search_tool)` to get search results
3. If no results: returns early with `"No sources found"` message
4. Calls `format_sources(results)` to build the formatted source string
5. Gets the prompt template from `create_analysis_prompt()`
6. Builds and runs the LangChain chain using pipe syntax:

```python
chain = prompt | llm | StrOutputParser()
analysis = chain.invoke({"claim": claim, "sources": formatted_sources})
```

7. Returns a dict:

```python
{
    "claim": claim,
    "analysis": analysis,        # full LLM output string
    "raw_sources": search_results,  # list of source dicts from Tavily
    "num_sources": len(search_results)
}
```

**Phase 2 done.** The backend pipeline is complete. You can test it by running:

```bash
uv run python -c "from detector import analyze_claim; print(analyze_claim('The earth is flat'))"
```

---

### PHASE 3: FRONTEND — `app.py`

This file contains the entire Streamlit UI in a single file.

---

#### Step 12: Build the Streamlit Frontend

**File:** `app.py`

**What it must contain:**

1. **Page config** — `st.set_page_config(page_title="Fake News Detector", layout="wide")`

2. **Title and description** — `st.title()` and a short `st.markdown()` description

3. **Input section** — `st.text_area()` for claim input with a placeholder like:

   ```bash
   "COVID-19 vaccines contain microchips that allow the government to track people"
   ```

4. **Three hardcoded example claims as buttons** — `st.button()` for instant testing:
   - Example 1: A known false claim
   - Example 2: A known true claim
   - Example 3: A misleading/partially true claim

5. **Submit button** — When clicked:
   - If input is empty: show `st.warning("Please enter a claim to analyze.")`
   - If input has text: show `st.spinner("Searching sources and analyzing...")` and call `analyze_claim(claim)`

6. **Results display:**
   - Parse the `VERDICT:` line from the analysis string using string parsing
   - Render a color-coded styled div using `st.markdown` with `unsafe_allow_html=True`:
     - TRUE → green background
     - FALSE → red background
     - MISLEADING → orange background
     - UNVERIFIED → yellow background
   - Two-column layout using `st.columns([2, 1])`:
     - Left column (2/3 width): full analysis text in a `st.markdown()` or `st.text()`
     - Right column (1/3 width): each source as an `st.expander()` card showing title, URL, and first 200 characters of content
   - `st.download_button()` exporting claim + full analysis as `fake_news_analysis.txt`

7. **Footer disclaimer:**

   ```text
   ⚠️ This tool uses AI to assist with fact-checking. Always cross-verify important claims
   with trusted sources like Snopes, PolitiFact, and AFP Fact Check.
   ```

**How to run:**

```bash
uv run streamlit run app.py
```

**Phase 3 done.** The full application is now running.

---

### PHASE 4: MANUAL EVALUATION

---

#### Step 13: Test with Real Claims

No automated testing. No pytest. No dataset downloads. Pure manual evaluation.

##### Where to Find Test News Articles (All Free)

**For fake/misleading news to test:**

- **Snopes.com** (<https://www.snopes.com>) — the gold standard. Every article is labeled True/False/Mixture. Copy the claim, paste into your app, see if it matches.
- **PolitiFact.com** (<https://www.politifact.com>) — political fact checks with detailed verdicts
- **LIAR dataset** (<https://huggingface.co/datasets/liar>) — 12,836 labeled statements
- **FakeNewsNet** (<https://github.com/KaiDMML/FakeNewsNet>) — full article text with labels

**For real news to test:**

- **Reuters** (<https://www.reuters.com>), **BBC** (<https://www.bbc.com>), **AP News** (<https://apnews.com>) — highly reliable, good for "TRUE" test cases

##### Quick Test Claims to Try Right Now

| # | Claim | Expected Verdict |
| --- | --- | --- |
| 1 | "The Great Wall of China is visible from space" | MISLEADING |
| 2 | "NASA's Artemis mission successfully launched" | TRUE |
| 3 | "Drinking bleach cures COVID-19" | FALSE |
| 4 | "India is the most populous country in the world as of 2023" | TRUE |

##### How to Evaluate

1. Open <https://www.snopes.com> in your browser
2. Browse their latest fact checks
3. Copy 10 headlines with known verdicts (mix of true, false, misleading)
4. Paste each into your running app at `http://localhost:8501`
5. Record in a simple table:

| # | Claim | Snopes Verdict | Our System Verdict | Match? | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | ... | False | FALSE | Yes | ... |
| 2 | ... | True | MISLEADING | No | Prompt issue or search returned weak sources |
| ... | ... | ... | ... | ... | ... |

6. For mismatches, investigate:
   - Were the Tavily search results relevant? If not, the search query may need tuning.
   - Did the LLM have enough evidence? If sources were sparse, the claim may be too obscure.
   - Did the LLM hallucinate? If so, the prompt needs stronger grounding instructions.

---

### PHASE 5: DOCUMENTATION AND DEPLOYMENT

---

#### Step 14: Write `README.md`

Must contain:

1. **Project title and description** — What it does, why it matters
2. **Architecture explanation** — Clearly call it a Search-Augmented Generation pipeline. Explain the three stages: Retrieval → Augmentation → Generation
3. **Architecture diagram** — The ASCII diagram from this project.md
4. **Setup instructions** — Exact `uv` commands from Phase 1 (Steps 1-6)
5. **API key instructions** — Where to get Groq and Tavily keys
6. **How to run** — `uv run streamlit run app.py`
7. **Example output** — What the user sees when they submit a claim

#### Step 15: GitHub Push

```bash
# Initialize git repo
git init

# Add all files
git add .

# Verify .env is NOT staged (check git status)
git status

# Commit
git commit -m "Initial commit: Explainable Fake News Detector"

# Create repo on GitHub and push
# (use gh repo create or create manually on github.com)
git remote add origin <your-repo-url>
git push -u origin main
```

**What gets pushed:** `.env.example`, `.gitignore`, `.python-version`, `pyproject.toml`, `uv.lock`, `detector.py`, `app.py`, `project.md`, `README.md`

**What never gets pushed:** `.env`, `.venv/`

#### Step 16: Deploy on Streamlit Community Cloud (Free)

1. Go to <https://streamlit.io/cloud>
2. Sign up with GitHub
3. Click "New app"
4. Connect your GitHub repo
5. Set main file to `app.py`
6. In "Secrets" panel, add:

```toml
GROQ_API_KEY = "your_actual_key"
TAVILY_API_KEY = "your_actual_key"
```

7. Deploy — your app is now live at `https://your-app-name.streamlit.app`

Note: For Streamlit Cloud deployment, update `initialize_tools()` to also check `st.secrets` as a fallback for API keys, since Streamlit Cloud uses secrets instead of `.env` files.

---

## Common Commands Reference

```bash
# Install all dependencies from lockfile
uv sync

# Run the Streamlit app
uv run streamlit run app.py

# Test the backend pipeline directly
uv run python -c "from detector import analyze_claim; print(analyze_claim('test claim'))"

# Add a new dependency
uv add package-name

# Update all dependencies
uv lock --upgrade
```

---

## Free Tier Limits

| Service | Free Tier Limit | What to Do When You Hit It |
| --- | --- | --- |
| Groq API | 30 req/min, ~14,400/day | Wait 1 minute and retry |
| Tavily API | 1,000 searches/month | Each analysis uses 2 searches, so ~500 analyses/month |
| Streamlit Cloud | Free hosting | N/A |

---

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `uv: command not found` | Restart terminal, or run `source ~/.zshrc` |
| Groq rate limit error | Wait 1 minute. You get 30 requests/minute on free tier |
| Tavily limit reached | You've used 1000 searches this month. Wait for next month. |
| Streamlit won't start | Make sure you're running `uv run streamlit run app.py` |
| Python not found | Run `uv python install 3.11` |
| `ModuleNotFoundError` | Run `uv sync` to install all dependencies |
| LLM returns malformed output | Check that the prompt template has all 6 required sections in the system message |

---

## Future Add-ons

These are enhancements to build after the core project is complete.

### Add-on A: Credibility Scoring Visual Meter

Parse the `CONFIDENCE:` field from the LLM output and display it as a visual gauge in the Streamlit UI. Map HIGH/MEDIUM/LOW to a color-coded progress bar (green/yellow/red).

**New dependency:** None

### Add-on B: History Tab

Store past analyses in `st.session_state` so users can compare multiple articles in a single session. Each analysis result gets appended to a list with claim, verdict, confidence, explanation, and timestamp.

**New dependency:** None

### Add-on C: Hindi Language Support

Add a dropdown in the UI for Hindi input. Translate Hindi text to English using `Helsinki-NLP/opus-mt-hi-en` from HuggingFace (runs locally), process through the existing pipeline, translate the explanation back. Huge differentiator for an Indian college project.

**New dependency:** `transformers`, `sentencepiece`

### Add-on D: Export to PDF Report

Generate a downloadable PDF for each analysis using `fpdf2` — verdict badge, confidence, full explanation, source table, timestamp.

**New dependency:** `fpdf2`

### Add-on E: Benchmark Tab

Run the system against 10-20 pre-labeled examples and display accuracy, precision, recall, F1 score in a table and confusion matrix chart.

**New dependency:** `scikit-learn`, `matplotlib`

---

## Project Status

- [ ] Phase 1: Environment Setup (Steps 1-6)
- [ ] Phase 2: Backend — detector.py (Steps 7-11)
- [ ] Phase 3: Frontend — app.py (Step 12)
- [ ] Phase 4: Manual Evaluation (Step 13)
- [ ] Phase 5: Documentation and Deployment (Steps 14-16)
- [ ] Future Add-on A: Credibility Scoring Visual Meter
- [ ] Future Add-on B: History Tab
- [ ] Future Add-on C: Hindi Language Support
- [ ] Future Add-on D: Export to PDF Report
- [ ] Future Add-on E: Benchmark Tab
