"""
YEA Business Query Fan-Out Tool
================================
A comprehensive query fan-out simulation tool for AEO (Answer Engine Optimization)
Based on Google's AI Mode patents and methodology.

Features:
- Light/White Theme for better visibility
- Full Data Export with table view
- Content Format Mapping
- CSV/JSON/Google Sheets export

Built for YEA Business - Digital Marketing & Video Production
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
from datetime import datetime
from typing import Optional, Dict, List
import time

# Page configuration
st.set_page_config(
    page_title="YEA Query Fan-Out Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - LIGHT/WHITE THEME
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ========== LIGHT THEME COLORS ========== */
    :root {
        --primary: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary: #f59e0b;
        --bg-main: #ffffff;
        --bg-card: #f8fafc;
        --bg-sidebar: #f1f5f9;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border: #e2e8f0;
        --border-light: #f1f5f9;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --purple: #8b5cf6;
        --pink: #ec4899;
        --cyan: #06b6d4;
    }
    
    /* ========== MAIN APP ========== */
    .stApp {
        background: var(--bg-main);
    }
    
    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary);
    }
    
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary);
    }
    
    /* ========== HEADERS ========== */
    .main-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: var(--text-secondary);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* ========== TEXT INPUT - VERY IMPORTANT FOR VISIBILITY ========== */
    .stTextInput > div > div > input {
        background: var(--bg-main) !important;
        border: 2px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* ========== TEXT AREA - VERY IMPORTANT FOR VISIBILITY ========== */
    .stTextArea > div > div > textarea {
        background: var(--bg-main) !important;
        border: 2px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* ========== SELECT BOX ========== */
    .stSelectbox > div > div {
        background: var(--bg-main) !important;
        border: 2px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
    }
    
    /* ========== LABELS ========== */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--purple) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Secondary buttons (example buttons) */
    .stButton > button[kind="secondary"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .stDownloadButton > button:hover {
        background: var(--primary) !important;
        color: white !important;
        border-color: var(--primary) !important;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-card);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
    
    /* ========== EXPANDERS ========== */
    div[data-testid="stExpander"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    
    div[data-testid="stExpander"] details summary {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        color: var(--primary);
    }
    
    /* ========== METRICS ========== */
    [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* ========== DATAFRAME/TABLE ========== */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* ========== ALERTS ========== */
    .stAlert {
        border-radius: 8px;
    }
    
    /* ========== MARKDOWN TEXT ========== */
    .stMarkdown {
        color: var(--text-primary);
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stMarkdown p {
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    .stMarkdown li {
        color: var(--text-primary);
    }
    
    .stMarkdown code {
        background: var(--bg-card);
        color: var(--primary);
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* ========== BAR CHART ========== */
    .stBarChart {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--border);
    }
    
    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-color: var(--primary) !important;
    }
    
    /* ========== FOOTER ========== */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid var(--border);
    }
    
    .footer strong {
        color: var(--text-primary);
    }
    
    /* ========== CUSTOM CARDS ========== */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .result-card:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CORE PROMPTS (Based on Google Patents) - WITH CONTENT FORMAT
# ============================================================================

QUERY_FANOUT_PROMPT_TEMPLATE = """
You are simulating Google's AI Mode query fan-out process based on the 
"Systems and methods for prompt-based query generation for diverse retrieval" 
patent (WO2024064249A1) and "Search with Stateful Chat" patent (US20240289407A1).

## ORIGINAL QUERY
"{query}"

## LANGUAGE
Generate all queries in: {language}

## MODE
{mode_instruction}

## CHAIN OF THOUGHT ANALYSIS

First, analyze the query step by step:

### Step 1: Intent Classification
- What is the user's primary intent?
- What decision are they trying to make?
- What information gaps do they have?

### Step 2: Entity Identification
- What is the main entity/topic?
- What related entities exist in the Knowledge Graph?
- What brands, products, or categories are relevant?

### Step 3: Implicit Needs
- What questions would naturally follow this query?
- What context would improve the answer?
- What comparisons might the user need?

### Step 4: User Journey Mapping
- Where is this user in their decision journey?
- What previous searches might have led here?
- What follow-up searches might occur?

## QUERY TYPE DEFINITIONS

Generate synthetic queries across these categories (based on Google patents):

1. **RELATED** - Semantically or categorically adjacent queries linked via entity relationships or taxonomy
   - Example: "best electric SUV" → "top rated electric crossovers"

2. **IMPLICIT** - Queries inferred from user intent that they didn't explicitly state
   - Example: "best electric SUV" → "EVs with longest range"

3. **COMPARATIVE** - Queries that compare products, entities, or options
   - Example: "best electric SUV" → "Rivian R1S vs Tesla Model X"

4. **REFORMULATION** - Lexical/syntactic rewrites maintaining core intent
   - Example: "best electric SUV" → "which electric SUV is best"

5. **ENTITY_EXPANDED** - Queries using Knowledge Graph entity relationships
   - Example: "best electric SUV" → "Model Y reviews", "Hyundai Ioniq 5 specs"

6. **PERSONALIZED** - Queries aligned to location, user context, or specific constraints
   - Example: "best electric SUV" → "EVs eligible for Malaysia rebate"

## INTENT CATEGORIES

Classify each query into one of these intents:
- **NAVIGATIONAL** - Looking for specific website/location
- **INFORMATIONAL** - Seeking knowledge/understanding
- **TRANSACTIONAL** - Ready to take action/purchase
- **COMPARATIVE** - Comparing options
- **EXPLORATORY** - Research phase, open-minded

## CONTENT FORMAT MAPPING

For each query, suggest the best content format for AEO teams:
- **BLOG_POST** - Deep dive educational article
- **FAQ_PAGE** - Quick answers to common questions
- **LANDING_PAGE** - Conversion-focused service page
- **COMPARISON_TABLE** - Side-by-side comparison content
- **VIDEO_CONTENT** - Explainer or testimonial video
- **TESTIMONIAL_PAGE** - Reviews and social proof
- **LOCAL_PAGE** - Location-specific content
- **HOW_TO_GUIDE** - Step-by-step tutorial
- **CASE_STUDY** - Success story with data

## OUTPUT REQUIREMENTS

Generate a JSON array with this EXACT structure:
```json
[
    {{
        "query": "the synthetic query text",
        "type": "RELATED|IMPLICIT|COMPARATIVE|REFORMULATION|ENTITY_EXPANDED|PERSONALIZED",
        "intent": "NAVIGATIONAL|INFORMATIONAL|TRANSACTIONAL|COMPARATIVE|EXPLORATORY",
        "content_format": "BLOG_POST|FAQ_PAGE|LANDING_PAGE|COMPARISON_TABLE|VIDEO_CONTENT|TESTIMONIAL_PAGE|LOCAL_PAGE|HOW_TO_GUIDE|CASE_STUDY",
        "reasoning": "Brief explanation of why Google would generate this query",
        "priority": "high|medium|low"
    }}
]
```

After the JSON array, provide:
1. **QUERY COUNT REASONING**: Why you chose this number of queries
2. **KEY THEMES**: The main themes/clusters identified
3. **CONTENT GAPS**: What content would be needed to cover these queries

## IMPORTANT RULES
- Each query must be unique and add value
- Cover diverse intents and angles
- Consider the {market} market context
- Include location-specific queries where relevant
- Balance between broad and specific queries
- Prioritize queries most likely to be generated by Google's AI
"""

AI_MODE_INSTRUCTION = """
AI MODE (Complex): Generate 25-35 synthetic queries covering the full spectrum of intents.
Google AI Mode performs extensive query fan-out for comprehensive answers.
Include queries across ALL categories with emphasis on comparative and implicit queries.
"""

AI_OVERVIEW_INSTRUCTION = """
AI OVERVIEW (Simple): Generate 12-18 synthetic queries for essential intents only.
AI Overviews use more focused query expansion.
Prioritize the most relevant related, implicit, and entity-expanded queries.
"""


# ============================================================================
# CONTENT ANALYSIS PROMPT
# ============================================================================

CONTENT_ANALYSIS_PROMPT = """
Analyze how well the following content covers the synthetic queries generated.

## CONTENT TO ANALYZE
{content}

## SYNTHETIC QUERIES TO CHECK
{queries}

## ANALYSIS REQUIREMENTS

For each synthetic query, determine:
1. **Coverage Score** (0-100): How well does the content address this query?
2. **Coverage Status**: 
   - COVERED (70-100): Content adequately addresses the query
   - PARTIAL (40-69): Content touches on this but needs expansion
   - GAP (0-39): Content doesn't address this query
3. **Specific Passage**: Quote the relevant passage if covered
4. **Improvement Suggestion**: How to better address this query

Output as JSON:
```json
[
    {{
        "query": "the synthetic query",
        "coverage_score": 85,
        "status": "COVERED|PARTIAL|GAP",
        "relevant_passage": "quoted passage or null",
        "improvement": "suggestion for improvement"
    }}
]
```

Also provide:
1. **OVERALL COVERAGE SCORE**: Percentage of queries well-covered
2. **TOP 5 GAPS**: Most important queries not covered
3. **QUICK WINS**: Easy improvements that would cover multiple queries
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def init_gemini(api_key: str) -> bool:
    """Initialize Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return False


def parse_json_from_response(response_text: str) -> List[Dict]:
    """Extract JSON array from LLM response."""
    try:
        # Try to find JSON array in the response
        json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', response_text)
        if json_match:
            json_str = json_match.group()
            # Clean up common issues
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing issue: {str(e)}")
    return []


def extract_sections(response_text: str) -> Dict[str, str]:
    """Extract additional sections from the response."""
    sections = {}
    
    # Query count reasoning
    count_match = re.search(r'QUERY COUNT REASONING[:\s]*(.*?)(?=KEY THEMES|$)', 
                           response_text, re.IGNORECASE | re.DOTALL)
    if count_match:
        sections['count_reasoning'] = count_match.group(1).strip()
    
    # Key themes
    themes_match = re.search(r'KEY THEMES[:\s]*(.*?)(?=CONTENT GAPS|$)', 
                            response_text, re.IGNORECASE | re.DOTALL)
    if themes_match:
        sections['themes'] = themes_match.group(1).strip()
    
    # Content gaps
    gaps_match = re.search(r'CONTENT GAPS[:\s]*(.*?)$', 
                          response_text, re.IGNORECASE | re.DOTALL)
    if gaps_match:
        sections['gaps'] = gaps_match.group(1).strip()
    
    return sections


def generate_fanout_queries(
    query: str,
    mode: str = "ai_mode",
    language: str = "English",
    market: str = "Malaysia/Singapore"
) -> Dict:
    """Generate synthetic fan-out queries using Gemini."""
    
    mode_instruction = AI_MODE_INSTRUCTION if mode == "ai_mode" else AI_OVERVIEW_INSTRUCTION
    
    prompt = QUERY_FANOUT_PROMPT_TEMPLATE.format(
        query=query,
        language=language,
        mode_instruction=mode_instruction,
        market=market
    )
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192
            )
        )
        
        response_text = response.text
        queries = parse_json_from_response(response_text)
        sections = extract_sections(response_text)
        
        return {
            "success": True,
            "queries": queries,
            "count_reasoning": sections.get('count_reasoning', ''),
            "themes": sections.get('themes', ''),
            "gaps": sections.get('gaps', ''),
            "raw_response": response_text,
            "total_count": len(queries)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "queries": [],
            "total_count": 0
        }


def analyze_content_coverage(content: str, queries: List[Dict]) -> Dict:
    """Analyze how well content covers the synthetic queries."""
    
    queries_text = json.dumps(queries, indent=2)
    prompt = CONTENT_ANALYSIS_PROMPT.format(
        content=content[:10000],  # Limit content length
        queries=queries_text
    )
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=4096
            )
        )
        
        analysis = parse_json_from_response(response.text)
        
        # Calculate overall stats
        if analysis:
            covered = sum(1 for a in analysis if a.get('status') == 'COVERED')
            partial = sum(1 for a in analysis if a.get('status') == 'PARTIAL')
            gaps = sum(1 for a in analysis if a.get('status') == 'GAP')
            avg_score = sum(a.get('coverage_score', 0) for a in analysis) / len(analysis)
        else:
            covered = partial = gaps = 0
            avg_score = 0
        
        return {
            "success": True,
            "analysis": analysis,
            "stats": {
                "covered": covered,
                "partial": partial,
                "gaps": gaps,
                "average_score": round(avg_score, 1)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis": []
        }


def get_format_emoji(format_type: str) -> str:
    """Get emoji for content format."""
    emojis = {
        "BLOG_POST": "📝",
        "FAQ_PAGE": "❓",
        "LANDING_PAGE": "🎯",
        "COMPARISON_TABLE": "⚖️",
        "VIDEO_CONTENT": "🎬",
        "TESTIMONIAL_PAGE": "⭐",
        "LOCAL_PAGE": "📍",
        "HOW_TO_GUIDE": "📋",
        "CASE_STUDY": "📊",
        "INFOGRAPHIC": "📈"
    }
    return emojis.get(format_type, "📄")


def get_intent_emoji(intent: str) -> str:
    """Get emoji for intent."""
    emojis = {
        "NAVIGATIONAL": "🧭",
        "INFORMATIONAL": "💡",
        "TRANSACTIONAL": "💰",
        "COMPARATIVE": "⚖️",
        "EXPLORATORY": "🔍"
    }
    return emojis.get(intent.upper(), "❓")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 YEA Query Fan-Out Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulate Google\'s AI Mode query expansion for AEO optimization</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your API key from https://aistudio.google.com/app/apikey"
        )
        
        st.markdown("---")
        
        mode = st.selectbox(
            "Analysis Mode",
            ["AI Mode (Complex)", "AI Overview (Simple)"],
            help="AI Mode generates more queries for comprehensive analysis"
        )
        
        language = st.selectbox(
            "Output Language",
            ["English", "Chinese (简体中文)", "Malay (Bahasa Malaysia)", "Mixed (EN/ZH)"],
            help="Language for generated queries"
        )
        
        market = st.selectbox(
            "Target Market",
            ["Malaysia/Singapore", "Southeast Asia", "Global", "Malaysia Only", "Singapore Only"],
            help="Market context for personalized queries"
        )
        
        st.markdown("---")
        
        st.markdown("### 📚 About")
        st.markdown("""
        This tool simulates Google's query fan-out process based on:
        - **WO2024064249A1** - Query generation patent
        - **US20240289407A1** - Stateful chat patent
        
        Built by **YEA Business** for AEO services.
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔗 Resources")
        st.markdown("""
        - [iPullRank AI Mode Guide](https://ipullrank.com/how-ai-mode-works)
        - [Google Patent](https://patents.google.com/patent/WO2024064249A1)
        - [WordLift Tool](https://wordlift.io/blog/en/query-fan-out-ai-search/)
        """)
    
    # Main content area
    tabs = st.tabs(["🔍 Query Fan-Out", "📊 Content Analysis", "📋 Full Data Export"])
    
    # Tab 1: Query Fan-Out
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_area(
                "Enter your target query",
                placeholder="e.g., TCM Johor Bahru, best digital marketing agency Malaysia",
                height=100,
                help="Enter the main query you want to analyze for fan-out"
            )
        
        with col2:
            st.markdown("### Quick Examples")
            example_queries = [
                "TCM Johor Bahru",
                "best SEO agency Malaysia",
                "video production company Singapore",
                "AI marketing tools for small business"
            ]
            for eq in example_queries:
                if st.button(eq, key=f"ex_{eq}"):
                    st.session_state['example_query'] = eq
                    st.rerun()
        
        # Check for example query
        if 'example_query' in st.session_state:
            query = st.session_state['example_query']
            del st.session_state['example_query']
        
        # Run analysis button
        if st.button("🚀 Generate Fan-Out Queries", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your Gemini API key in the sidebar")
            elif not query:
                st.error("Please enter a query to analyze")
            else:
                if init_gemini(api_key):
                    with st.spinner("Generating synthetic queries... This may take 15-30 seconds"):
                        mode_key = "ai_mode" if "Complex" in mode else "ai_overview"
                        lang_map = {
                            "English": "English",
                            "Chinese (简体中文)": "Simplified Chinese (简体中文)",
                            "Malay (Bahasa Malaysia)": "Bahasa Malaysia",
                            "Mixed (EN/ZH)": "Mixed English and Chinese"
                        }
                        
                        result = generate_fanout_queries(
                            query=query,
                            mode=mode_key,
                            language=lang_map.get(language, "English"),
                            market=market
                        )
                        
                        if result["success"]:
                            st.session_state['last_result'] = result
                            st.session_state['last_query'] = query
                            st.success(f"✅ Generated {result['total_count']} synthetic queries!")
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Display results
        if 'last_result' in st.session_state and st.session_state['last_result']['success']:
            result = st.session_state['last_result']
            queries = result['queries']
            
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Stats row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", result['total_count'])
            
            with col2:
                type_counts = {}
                for q in queries:
                    qtype = q.get('type', 'OTHER')
                    type_counts[qtype] = type_counts.get(qtype, 0) + 1
                st.metric("Query Types", len(type_counts))
            
            with col3:
                high_priority = sum(1 for q in queries if q.get('priority') == 'high')
                st.metric("High Priority", high_priority)
            
            with col4:
                intent_counts = {}
                for q in queries:
                    intent = q.get('intent', 'OTHER')
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
                st.metric("Intent Types", len(intent_counts))
            
            # Query type distribution
            st.markdown("### 📈 Query Type Distribution")
            
            type_data = []
            for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                type_data.append({"Type": qtype, "Count": count})
            
            if type_data:
                df_types = pd.DataFrame(type_data)
                st.bar_chart(df_types.set_index('Type'))
            
            # Display queries by intent
            st.markdown("### 🔍 Queries by Intent")
            
            for intent in sorted(intent_counts.keys()):
                intent_queries = [q for q in queries if q.get('intent', '').upper() == intent.upper()]
                with st.expander(f"{get_intent_emoji(intent)} {intent} ({len(intent_queries)} queries)", expanded=False):
                    for q in intent_queries:
                        fmt = q.get('content_format', 'BLOG_POST')
                        priority = q.get('priority', 'medium')
                        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                        
                        st.markdown(f"""
                        **{q.get('query', '')}**  
                        {priority_emoji} {priority.upper()} | {get_format_emoji(fmt)} {fmt} | `{q.get('type', '')}`  
                        _{q.get('reasoning', '')}_
                        """)
                        st.markdown("---")
            
            # Additional insights
            if result.get('count_reasoning') or result.get('themes') or result.get('gaps'):
                st.markdown("### 💡 Analysis Insights")
                
                if result.get('count_reasoning'):
                    with st.expander("Query Count Reasoning"):
                        st.markdown(result['count_reasoning'])
                
                if result.get('themes'):
                    with st.expander("Key Themes Identified"):
                        st.markdown(result['themes'])
                
                if result.get('gaps'):
                    with st.expander("Content Gap Analysis"):
                        st.markdown(result['gaps'])
    
    # Tab 2: Content Analysis
    with tabs[1]:
        st.markdown("## 📊 Content Coverage Analysis")
        st.markdown("Analyze how well your content covers the generated fan-out queries")
        
        content_input = st.text_area(
            "Paste your content here",
            placeholder="Paste the content of your page/article to analyze coverage...",
            height=300
        )
        
        if st.button("🔍 Analyze Coverage", type="primary"):
            if 'last_result' not in st.session_state:
                st.error("Please generate fan-out queries first (in the Query Fan-Out tab)")
            elif not content_input:
                st.error("Please paste content to analyze")
            elif not api_key:
                st.error("Please enter your Gemini API key")
            else:
                with st.spinner("Analyzing content coverage..."):
                    analysis = analyze_content_coverage(
                        content_input,
                        st.session_state['last_result']['queries']
                    )
                    
                    if analysis["success"]:
                        st.session_state['coverage_analysis'] = analysis
                        st.success("Analysis complete!")
                    else:
                        st.error(f"Error: {analysis.get('error', 'Unknown error')}")
        
        # Display coverage results
        if 'coverage_analysis' in st.session_state and st.session_state['coverage_analysis']['success']:
            analysis = st.session_state['coverage_analysis']
            stats = analysis['stats']
            
            st.markdown("### 📈 Coverage Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Score", f"{stats['average_score']}%")
            with col2:
                st.metric("✅ Covered", stats['covered'])
            with col3:
                st.metric("⚠️ Partial", stats['partial'])
            with col4:
                st.metric("❌ Gaps", stats['gaps'])
            
            # Coverage breakdown
            st.markdown("### 📋 Detailed Coverage")
            
            for item in analysis['analysis']:
                status = item.get('status', 'UNKNOWN')
                status_emoji = {"COVERED": "✅", "PARTIAL": "⚠️", "GAP": "❌"}.get(status, "❓")
                score = item.get('coverage_score', 0)
                
                with st.expander(f"{status_emoji} {item.get('query', 'N/A')} - Score: {score}%"):
                    st.markdown(f"**Status:** {status}")
                    
                    if item.get('relevant_passage'):
                        st.markdown(f"**Relevant Passage:** _{item['relevant_passage']}_")
                    
                    if item.get('improvement'):
                        st.markdown(f"**Improvement Suggestion:** {item['improvement']}")
    
    # Tab 3: Full Data Export (NEW - like screenshot)
    with tabs[2]:
        st.markdown("## 📋 Full Data Export")
        
        if 'last_result' in st.session_state and st.session_state['last_result']['success']:
            result = st.session_state['last_result']
            queries = result['queries']
            
            # Create DataFrame
            df = pd.DataFrame(queries)
            
            # Reorder columns for better display
            column_order = ['query', 'type', 'intent', 'content_format', 'reasoning', 'priority']
            available_columns = [col for col in column_order if col in df.columns]
            df = df[available_columns]
            
            st.markdown("### 📁 Complete Query Data")
            
            # Display the dataframe (like the screenshot)
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
                column_config={
                    "query": st.column_config.TextColumn("Query", width="large"),
                    "type": st.column_config.TextColumn("Type", width="medium"),
                    "intent": st.column_config.TextColumn("Intent", width="medium"),
                    "content_format": st.column_config.TextColumn("Content Format", width="medium"),
                    "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
                    "priority": st.column_config.TextColumn("Priority", width="small"),
                }
            )
            
            # Export options (like the screenshot)
            st.markdown("### 📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📄 CSV",
                    csv,
                    f"fanout_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    "📋 JSON",
                    json_str,
                    f"fanout_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Google Sheets format (simpler columns)
                sheets_df = df[['query', 'intent', 'content_format', 'priority']].copy()
                sheets_csv = sheets_df.to_csv(index=False)
                st.download_button(
                    "📊 Google Sheets",
                    sheets_csv,
                    f"fanout_sheets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**By Query Type:**")
                type_summary = df['type'].value_counts().to_frame()
                type_summary.columns = ['Count']
                st.dataframe(type_summary, use_container_width=True)
            
            with col2:
                st.markdown("**By Content Format:**")
                if 'content_format' in df.columns:
                    format_summary = df['content_format'].value_counts().to_frame()
                    format_summary.columns = ['Count']
                    st.dataframe(format_summary, use_container_width=True)
            
            # AI Insights
            if result.get('themes') or result.get('gaps') or result.get('count_reasoning'):
                st.markdown("### 💡 AI Insights")
                
                if result.get('themes'):
                    with st.expander("Key Themes"):
                        st.markdown(result['themes'])
                
                if result.get('gaps'):
                    with st.expander("Content Gaps"):
                        st.markdown(result['gaps'])
                
                if result.get('count_reasoning'):
                    with st.expander("Query Count Reasoning"):
                        st.markdown(result['count_reasoning'])
        
        else:
            st.info("👆 Generate queries first in the 'Query Fan-Out' tab to see and export data")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>YEA Query Fan-Out Tool</strong> | Built for AEO Optimization</p>
        <p>Based on Google AI Mode Patents | Powered by Gemini</p>
        <p>© 2024 YEA Business - Digital Marketing & Video Production</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
