"""
YEA Business Query Fan-Out Tool
================================
A comprehensive query fan-out simulation tool for AEO (Answer Engine Optimization)
Based on Google's AI Mode patents and methodology.

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

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: #334155;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, var(--primary) 0%, #7c3aed 100%);
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
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .query-type-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-related { background: #3b82f6; color: white; }
    .badge-implicit { background: #8b5cf6; color: white; }
    .badge-comparative { background: #f59e0b; color: black; }
    .badge-reformulation { background: #10b981; color: white; }
    .badge-entity { background: #ec4899; color: white; }
    .badge-personalized { background: #06b6d4; color: black; }
    
    .stTextInput > div > div > input {
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stTextArea > div > div > textarea {
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
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
        transform: translateY(-2px);
    }
    
    .intent-tag {
        background: rgba(37, 99, 235, 0.2);
        color: var(--primary);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    .reasoning-text {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    div[data-testid="stExpander"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CORE PROMPTS (Based on Google Patents)
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
   - Trigger: Co-occurrence patterns, topical proximity in Knowledge Graph
   - Example: "best electric SUV" → "top rated electric crossovers"

2. **IMPLICIT** - Queries inferred from user intent that they didn't explicitly state
   - Trigger: LLM inference from phrasing, ambiguity, user behavior
   - Example: "best electric SUV" → "EVs with longest range"

3. **COMPARATIVE** - Queries that compare products, entities, or options
   - Trigger: Decision-making or choice scenarios
   - Example: "best electric SUV" → "Rivian R1S vs Tesla Model X"

4. **REFORMULATION** - Lexical/syntactic rewrites maintaining core intent
   - Trigger: Standard query expansion
   - Example: "best electric SUV" → "which electric SUV is best"

5. **ENTITY_EXPANDED** - Queries using Knowledge Graph entity relationships
   - Trigger: Entity crosswalks to broader/narrower equivalents
   - Example: "best electric SUV" → "Model Y reviews", "Hyundai Ioniq 5 specs"

6. **PERSONALIZED** - Queries aligned to location, user context, or specific constraints
   - Trigger: Location, time, or demographic signals
   - Example: "best electric SUV" → "EVs eligible for Malaysia rebate"

## OUTPUT REQUIREMENTS

Generate a JSON array with this EXACT structure:
```json
[
    {{
        "query": "the synthetic query text",
        "type": "RELATED|IMPLICIT|COMPARATIVE|REFORMULATION|ENTITY_EXPANDED|PERSONALIZED",
        "intent": "informational|transactional|navigational|comparative|exploratory",
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


def get_query_type_color(query_type: str) -> str:
    """Get badge color for query type."""
    colors = {
        "RELATED": "#3b82f6",
        "IMPLICIT": "#8b5cf6",
        "COMPARATIVE": "#f59e0b",
        "REFORMULATION": "#10b981",
        "ENTITY_EXPANDED": "#ec4899",
        "PERSONALIZED": "#06b6d4"
    }
    return colors.get(query_type, "#6b7280")


def get_intent_color(intent: str) -> str:
    """Get color for intent type."""
    colors = {
        "informational": "#3b82f6",
        "transactional": "#10b981",
        "navigational": "#f59e0b",
        "comparative": "#8b5cf6",
        "exploratory": "#ec4899"
    }
    return colors.get(intent.lower(), "#6b7280")


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
    tabs = st.tabs(["🔍 Query Fan-Out", "📊 Content Analysis", "📈 Results History"])
    
    # Tab 1: Query Fan-Out
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_area(
                "Enter your target query",
                placeholder="e.g., best digital marketing agency in Malaysia",
                height=100,
                help="Enter the main query you want to analyze for fan-out"
            )
        
        with col2:
            st.markdown("### Quick Examples")
            example_queries = [
                "best SEO agency Malaysia",
                "video production company Singapore",
                "AI marketing tools for small business",
                "how to improve website ranking"
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
                            st.success(f"Generated {result['total_count']} synthetic queries!")
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
                st.metric("Mode", "AI Mode" if "ai_mode" in str(mode).lower() else "Overview")
            
            # Query type distribution
            st.markdown("### 📈 Query Type Distribution")
            
            type_data = []
            for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                type_data.append({"Type": qtype, "Count": count})
            
            if type_data:
                df_types = pd.DataFrame(type_data)
                st.bar_chart(df_types.set_index('Type'))
            
            # Display queries
            st.markdown("### 🔍 Generated Queries")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_type = st.multiselect(
                    "Filter by Type",
                    options=list(type_counts.keys()),
                    default=list(type_counts.keys())
                )
            with col2:
                filter_priority = st.multiselect(
                    "Filter by Priority",
                    options=["high", "medium", "low"],
                    default=["high", "medium", "low"]
                )
            
            # Display filtered queries
            for i, q in enumerate(queries):
                qtype = q.get('type', 'OTHER')
                priority = q.get('priority', 'medium')
                
                if qtype not in filter_type or priority not in filter_priority:
                    continue
                
                with st.expander(f"**{q.get('query', 'N/A')}**", expanded=(priority == 'high' and i < 5)):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Type:** `{qtype}`")
                    with col2:
                        st.markdown(f"**Intent:** `{q.get('intent', 'N/A')}`")
                    with col3:
                        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                        st.markdown(f"**Priority:** {priority_emoji} {priority}")
                    
                    st.markdown(f"**Reasoning:** {q.get('reasoning', 'N/A')}")
            
            # Additional insights
            if result.get('count_reasoning'):
                st.markdown("### 💡 Analysis Insights")
                
                with st.expander("Query Count Reasoning"):
                    st.markdown(result['count_reasoning'])
                
                if result.get('themes'):
                    with st.expander("Key Themes Identified"):
                        st.markdown(result['themes'])
                
                if result.get('gaps'):
                    with st.expander("Content Gap Analysis"):
                        st.markdown(result['gaps'])
            
            # Export options
            st.markdown("### 📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                df = pd.DataFrame(queries)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv,
                    f"fanout_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    "📋 Download JSON",
                    json_str,
                    f"fanout_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Generate markdown report
                md_report = f"""# Query Fan-Out Analysis Report

## Original Query
{st.session_state.get('last_query', 'N/A')}

## Analysis Settings
- Mode: {mode}
- Language: {language}
- Market: {market}
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Queries: {result['total_count']}
- Query Types: {len(type_counts)}

## Query Type Distribution
{chr(10).join([f"- {t}: {c}" for t, c in type_counts.items()])}

## Generated Queries

{chr(10).join([f"### {i+1}. {q.get('query', 'N/A')}{chr(10)}- Type: {q.get('type', 'N/A')}{chr(10)}- Intent: {q.get('intent', 'N/A')}{chr(10)}- Priority: {q.get('priority', 'N/A')}{chr(10)}- Reasoning: {q.get('reasoning', 'N/A')}{chr(10)}" for i, q in enumerate(queries)])}

## Insights

### Count Reasoning
{result.get('count_reasoning', 'N/A')}

### Key Themes
{result.get('themes', 'N/A')}

### Content Gaps
{result.get('gaps', 'N/A')}

---
Generated by YEA Query Fan-Out Tool
"""
                st.download_button(
                    "📝 Download Report",
                    md_report,
                    f"fanout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    use_container_width=True
                )
    
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
                st.metric("Covered", stats['covered'], help="Queries well-covered by content")
            with col3:
                st.metric("Partial", stats['partial'], help="Queries partially covered")
            with col4:
                st.metric("Gaps", stats['gaps'], help="Queries not covered")
            
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
    
    # Tab 3: Results History (placeholder for future database integration)
    with tabs[2]:
        st.markdown("## 📈 Results History")
        st.info("This feature will store your analysis history for tracking improvements over time. Coming soon!")
        
        if 'last_result' in st.session_state:
            st.markdown("### Last Analysis")
            st.markdown(f"**Query:** {st.session_state.get('last_query', 'N/A')}")
            st.markdown(f"**Queries Generated:** {st.session_state['last_result']['total_count']}")
            st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>YEA Query Fan-Out Tool</strong> | Built for AEO Optimization</p>
        <p>Based on Google AI Mode Patents | Powered by Gemini 2.5 Pro</p>
        <p>© 2024 YEA Business - Digital Marketing & Video Production</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
