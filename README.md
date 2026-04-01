# Explainable Fake News Detection using LLMs with Multi-Source Reasoning

An AI-powered fact-checking system that analyzes news claims, searches multiple sources, and delivers verdicts with detailed explanations and source citations.

## What It Does

Paste any news headline or claim, and the system will:

1. **Search** the live web for related reporting from multiple sources
2. **Cross-reference** the claim against what sources actually say
3. **Generate** a verdict (TRUE / FALSE / MISLEADING / UNVERIFIED) with confidence level
4. **Explain** the reasoning with inline source citations

## Architecture

This system implements a **Search-Augmented Generation (SAG)** pipeline — a modern, web-based variant of RAG where retrieval happens from the live internet instead of a local vector database.

```
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
│  └────────────────────────┬─────────────────────┘   │
│                           │                          │
│                           ▼                          │
│  ┌──────────────────────────────────────────────┐   │
│  │  AUGMENTATION                                │   │
│  │  format_sources(results)                     │   │
│  │  └─ Format each source as numbered block     │   │
│  │     SOURCE 1: Title | URL | Content          │   │
│  │     SOURCE 2: Title | URL | Content          │   │
│  │     ...                                      │   │
│  └────────────────────────┬─────────────────────┘   │
│                           │                          │
│                           ▼                          │
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

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| Package Manager | uv | Free |
| Python | 3.11 | Free |
| LLM | Groq API (llama-3.3-70b-versatile) | Free (30 req/min) |
| Web Search | Tavily API | Free (1000 searches/month) |
| Orchestration | LangChain | Free |
| Frontend | Streamlit | Free |
| Deployment | Streamlit Community Cloud | Free |

## Setup

### Prerequisites

- macOS/Linux terminal
- Internet connection
- API keys (free)

### Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then verify:

```bash
uv --version
```

### Step 2: Install Python 3.11

```bash
uv python install 3.11
```

### Step 3: Clone and Setup Project

```bash
git clone <your-repo-url>
cd fake-news-detection

# Install dependencies
uv sync
```

### Step 4: Get API Keys

**Groq (LLM API):**
1. Go to https://console.groq.com
2. Sign up with Google or GitHub (no credit card)
3. Go to "API Keys" → "Create API Key"
4. Copy the key

**Tavily (Web Search):**
1. Go to https://tavily.com
2. Sign up (no credit card)
3. Get your API key from the dashboard

### Step 5: Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your keys
# GROQ_API_KEY=your_groq_key_here
# TAVILY_API_KEY=your_tavily_key_here
```

## Running the App

```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`

## Example Usage

1. **Paste a claim:**
   ```
   "The Great Wall of China is visible from space"
   ```

2. **Click "Analyze Claim"**

3. **View the results:**
   - Color-coded verdict badge (green/red/orange/yellow)
   - Full analysis with evidence summary
   - Reasoning with inline source citations
   - Red flags detected
   - Sources consulted with URLs

4. **Download the report** as a `.txt` file

## Example Output

```
VERDICT: MISLEADING

CONFIDENCE: HIGH

EVIDENCE SUMMARY:
- The Great Wall is not visible from space with the naked eye (Source 1, Source 3)
- Astronauts have confirmed this misconception (Source 2)
- The wall is too narrow (~6m wide) to be seen from low Earth orbit

REASONING:
The claim that the Great Wall of China is visible from space is a common
misconception. According to NASA astronauts and multiple sources (Source 1),
the Great Wall is not visible from space with the naked eye. While the wall
is very long (~21,000 km), its width is only about 6 meters, making it
impossible to distinguish from surrounding terrain at orbital altitudes.
This myth likely originated from misquoted statements and has been debunked
by numerous astronauts including Neil Armstrong and Chris Hadfield (Source 2).

RED FLAGS:
- Common misconception repeated without verification
- No credible scientific source supports this claim

SOURCES CONSULTED:
- https://www.nasa.gov/...
- https://www.space.com/...
- https://www.snopes.com/...
```

## Project Structure

```
fake-news-detection/
├── .env              # API keys (never push)
├── .env.example      # Placeholder keys
├── .gitignore        # Git exclusions
├── .python-version   # Python 3.11
├── pyproject.toml    # Dependencies
├── uv.lock           # Lockfile
├── detector.py       # Backend pipeline
├── app.py            # Streamlit frontend
├── project.md        # Detailed project plan
└── README.md         # This file
```

## Deployment

### Streamlit Community Cloud (Free)

1. Push your code to GitHub (ensure `.env` is NOT included)
2. Go to https://streamlit.io/cloud
3. Sign up with GitHub
4. Click "New app" → Connect your repo
5. Set main file to `app.py`
6. Add secrets in the dashboard:
   ```toml
   GROQ_API_KEY = "your_actual_key"
   TAVILY_API_KEY = "your_actual_key"
   ```
7. Deploy

Your app will be live at `https://your-app-name.streamlit.app`

## Free Tier Limits

| Service | Limit | Notes |
|---------|-------|-------|
| Groq API | 30 req/min, 14,400/day | Wait 1 min if rate limited |
| Tavily API | 1,000 searches/month | Each analysis uses 2 searches |
| Streamlit Cloud | Free hosting | No limits |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` | Restart terminal or run `source ~/.zshrc` |
| Groq rate limit | Wait 1 minute and retry |
| Tavily limit reached | Wait for next month (1000 searches reset) |
| ModuleNotFoundError | Run `uv sync` |
| Streamlit won't start | Use `uv run streamlit run app.py` |

## Disclaimer

This tool uses AI to assist with fact-checking. Always cross-verify important claims with trusted sources like [Snopes](https://www.snopes.com), [PolitiFact](https://www.politifact.com), and [AFP Fact Check](https://factcheck.afp.com).

## License

MIT License
