"""
app.py — Streamlit Frontend for Explainable Fake News Detection

This module provides the web UI for the fake news detection system.
Users can paste news claims and receive structured fact-checking analysis.
"""

import streamlit as st
from detector import analyze_claim


# --- Page Config ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)


# --- Title and Description ---
st.title("🔍 Explainable Fake News Detector")
st.markdown("""
Paste a news headline or claim below, and the system will:
1. **Search** multiple sources on the web
2. **Cross-reference** what the claim says against the sources
3. **Generate** a verdict with a detailed explanation and source citations
""")


# --- Helper Functions ---

def parse_verdict(analysis: str) -> str:
    """
    Parses the VERDICT line from the LLM analysis output.

    Args:
        analysis: Full analysis string from the LLM

    Returns:
        str: The verdict (TRUE, FALSE, MISLEADING, UNVERIFIED) or 'UNKNOWN'
    """
    for line in analysis.split('\n'):
        line = line.strip().upper()
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
            # Handle cases like "VERDICT: [TRUE]" or "VERDICT: TRUE"
            verdict = verdict.strip("[]").strip()
            if verdict in ["TRUE", "FALSE", "MISLEADING", "UNVERIFIED"]:
                return verdict
    return "UNKNOWN"


def get_verdict_color(verdict: str) -> str:
    """
    Returns the background color for a given verdict.

    Args:
        verdict: The verdict string (TRUE, FALSE, MISLEADING, UNVERIFIED)

    Returns:
        str: CSS color value
    """
    colors = {
        "TRUE": "#28a745",        # Green
        "FALSE": "#dc3545",       # Red
        "MISLEADING": "#fd7e14",  # Orange
        "UNVERIFIED": "#ffc107",  # Yellow
        "UNKNOWN": "#6c757d"      # Gray
    }
    return colors.get(verdict, colors["UNKNOWN"])


def render_verdict_badge(verdict: str):
    """
    Renders a color-coded verdict badge using HTML/CSS.

    Args:
        verdict: The verdict string
    """
    color = get_verdict_color(verdict)
    html = f"""
    <div style="
        background-color: {color};
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        VERDICT: {verdict}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# --- Sidebar with Example Claims ---
st.sidebar.header("📌 Quick Test Claims")
st.sidebar.markdown("Click any example to test instantly:")

example_claims = [
    ("The Great Wall of China is visible from space", "MISLEADING"),
    ("NASA's Artemis mission successfully launched", "TRUE"),
    ("Drinking bleach cures COVID-19", "FALSE"),
]

for claim, expected in example_claims:
    if st.sidebar.button(f"📋 {claim[:40]}...", key=f"example_{hash(claim)}"):
        st.session_state["claim_input"] = claim
        st.rerun()


# --- Main Input Section ---
st.header("📝 Enter a Claim to Fact-Check")

# Text area for claim input (uses session state for example button clicks)
default_claim = st.session_state.get("claim_input", "")
claim_input = st.text_area(
    "Paste a news headline, claim, or article excerpt:",
    key="claim_input",
    height=120,
    placeholder="Example: COVID-19 vaccines contain microchips that allow the government to track people"
)

# Clear session state after using
if "claim_input" in st.session_state:
    del st.session_state["claim_input"]


# --- Submit Button ---
col_submit, col_clear = st.columns([1, 1])

with col_submit:
    analyze_button = st.button("🔍 Analyze Claim", type="primary", use_container_width=True)

with col_clear:
    clear_button = st.button("🗑️ Clear", use_container_width=True)
    if clear_button:
        st.rerun()


# --- Processing and Results ---
if analyze_button:
    if not claim_input.strip():
        st.warning("⚠️ Please enter a claim to analyze.")
    else:
        # Show spinner while processing
        with st.spinner("🔍 Searching sources and analyzing... This may take 10-20 seconds."):
            result = analyze_claim(claim_input.strip())

        # Check if analysis succeeded
        if result["num_sources"] == 0:
            st.error("❌ No sources found for this claim. Try a different search term.")
        else:
            # Parse verdict
            verdict = parse_verdict(result["analysis"])

            # Display verdict badge
            st.markdown("---")
            render_verdict_badge(verdict)

            # Two-column layout: Analysis (left) | Sources (right)
            col_analysis, col_sources = st.columns([2, 1])

            with col_analysis:
                st.subheader("📊 Full Analysis")
                st.markdown(result["analysis"])

            with col_sources:
                st.subheader("📚 Sources Consulted")
                st.caption(f"{result['num_sources']} sources used")

                for i, source in enumerate(result["raw_sources"], 1):
                    title = source.get("title", "No title")
                    url = source.get("url", "#")
                    content = source.get("content", "No content available")

                    # Truncate content to first 200 characters
                    content_preview = content[:200] + "..." if len(content) > 200 else content

                    with st.expander(f"**Source {i}:** {title[:50]}{'...' if len(title) > 50 else ''}"):
                        st.markdown(f"**URL:** [{url}]({url})")
                        st.markdown(f"**Preview:**")
                        st.caption(content_preview)

            # Download button for report
            st.markdown("---")
            st.subheader("📥 Download Report")

            report_content = f"""FAKE NEWS ANALYSIS REPORT
{'=' * 50}

CLAIM ANALYZED:
{result['claim']}

{'=' * 50}

ANALYSIS:
{result['analysis']}

{'=' * 50}

SOURCES CONSULTED ({result['num_sources']} total):
"""

            for i, source in enumerate(result["raw_sources"], 1):
                report_content += f"""
Source {i}:
  Title: {source.get('title', 'No title')}
  URL: {source.get('url', 'No URL')}
  Content: {source.get('content', 'No content')}
"""

            report_content += f"""
{'=' * 50}
Generated by Explainable Fake News Detector
"""

            st.download_button(
                label="📄 Download Analysis Report (.txt)",
                data=report_content,
                file_name="fake_news_analysis.txt",
                mime="text/plain",
                use_container_width=True
            )


# --- Footer Disclaimer ---
st.markdown("---")
st.markdown("""
<div style="
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    font-size: 14px;
">
⚠️ <strong>Disclaimer:</strong> This tool uses AI to assist with fact-checking. Always cross-verify important claims
with trusted sources like <a href="https://www.snopes.com" target="_blank">Snopes</a>,
<a href="https://www.politifact.com" target="_blank">PolitiFact</a>, and
<a href="https://factcheck.afp.com" target="_blank">AFP Fact Check</a>.
</div>
""", unsafe_allow_html=True)
