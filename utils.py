"""
YEA Query Fan-Out Advanced Utilities
=====================================
Additional tools for semantic analysis, batch processing, and SERP integration.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SyntheticQuery:
    """Represents a single synthetic query from fan-out."""
    query: str
    query_type: str
    intent: str
    reasoning: str
    priority: str
    coverage_score: Optional[float] = None
    ranking_position: Optional[int] = None


@dataclass
class FanOutResult:
    """Complete fan-out analysis result."""
    original_query: str
    mode: str
    language: str
    market: str
    queries: List[SyntheticQuery]
    count_reasoning: str
    themes: str
    gaps: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "mode": self.mode,
            "language": self.language,
            "market": self.market,
            "queries": [vars(q) for q in self.queries],
            "count_reasoning": self.count_reasoning,
            "themes": self.themes,
            "gaps": self.gaps,
            "timestamp": self.timestamp.isoformat(),
            "total_queries": len(self.queries)
        }


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

BATCH_FANOUT_PROMPT = """
Analyze multiple queries and generate fan-out queries for each.

## QUERIES TO ANALYZE
{queries}

## LANGUAGE
{language}

## MARKET CONTEXT
{market}

For each query, generate 8-12 essential synthetic queries covering:
- Related queries (semantic adjacency)
- Implicit queries (unstated needs)
- Comparative queries (vs alternatives)
- Entity-expanded queries (specific brands/locations)

Output as JSON:
```json
{{
    "results": [
        {{
            "original_query": "query 1",
            "synthetic_queries": [
                {{"query": "...", "type": "...", "intent": "...", "priority": "..."}}
            ]
        }}
    ]
}}
```
"""


CONTENT_OPTIMIZATION_PROMPT = """
Based on the content gaps identified, generate specific content recommendations.

## ORIGINAL QUERY
{query}

## CONTENT GAPS (queries not covered)
{gaps}

## EXISTING CONTENT SUMMARY
{content_summary}

Generate actionable recommendations:

1. **New Sections to Add**: Specific sections/headings to add
2. **FAQ Questions**: Questions to add to FAQ section
3. **Passage Improvements**: Specific passages to modify
4. **Internal Links**: Related content to link to
5. **Entity Mentions**: Entities to include for Knowledge Graph

Output as JSON:
```json
{{
    "new_sections": [
        {{"heading": "...", "content_brief": "...", "covers_queries": [...]}}
    ],
    "faq_questions": [
        {{"question": "...", "answer_brief": "...", "covers_query": "..."}}
    ],
    "passage_improvements": [
        {{"original": "...", "improved": "...", "reasoning": "..."}}
    ],
    "internal_links": [
        {{"anchor_text": "...", "target_topic": "...", "reasoning": "..."}}
    ],
    "entity_mentions": [
        {{"entity": "...", "context": "...", "importance": "high|medium|low"}}
    ]
}}
```
"""


COMPETITOR_ANALYSIS_PROMPT = """
Analyze competitor content against the synthetic queries.

## TARGET QUERY
{query}

## SYNTHETIC QUERIES
{synthetic_queries}

## COMPETITOR CONTENT
{competitor_content}

## YOUR CONTENT
{your_content}

Analyze:
1. Which synthetic queries does the competitor cover that you don't?
2. Which queries do you cover better?
3. What unique angles does the competitor have?
4. What opportunities are they missing?

Output as JSON:
```json
{{
    "competitor_advantages": [
        {{"query": "...", "their_coverage": "...", "recommendation": "..."}}
    ],
    "your_advantages": [
        {{"query": "...", "your_strength": "..."}}
    ],
    "opportunities": [
        {{"query": "...", "why_opportunity": "...", "action": "..."}}
    ],
    "content_gaps_both_miss": [
        {{"query": "...", "first_mover_opportunity": "..."}}
    ]
}}
```
"""


PASSAGE_OPTIMIZATION_PROMPT = """
Optimize specific content passages for better AI Mode retrieval.

## TARGET SYNTHETIC QUERY
{query}

## CURRENT PASSAGE
{passage}

## OPTIMIZATION GOALS
Based on Google's AI Mode patents, optimize for:
1. Semantic completeness - passage should answer query standalone
2. Entity clarity - clear entity references
3. Factual density - specific stats/facts
4. Readability - clear, scannable structure

Generate optimized passage and explain changes:

```json
{{
    "original_passage": "...",
    "optimized_passage": "...",
    "changes_made": [
        {{"change": "...", "reasoning": "..."}}
    ],
    "semantic_score_improvement": "estimated % improvement",
    "entities_added": ["..."],
    "facts_added": ["..."]
}}
```
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_query_similarity(query1: str, query2: str, model=None) -> float:
    """
    Calculate semantic similarity between two queries using Gemini.
    Returns score from 0-100.
    """
    if model is None:
        model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Rate the semantic similarity between these two search queries on a scale of 0-100.
    
    Query 1: "{query1}"
    Query 2: "{query2}"
    
    Consider:
    - Do they have the same intent?
    - Would the same content satisfy both?
    - Are they asking about the same topic?
    
    Respond with ONLY a number between 0 and 100.
    """
    
    try:
        response = model.generate_content(prompt)
        score = float(re.search(r'\d+', response.text).group())
        return min(100, max(0, score))
    except:
        return 0.0


def cluster_queries(queries: List[str], threshold: float = 70.0) -> List[List[str]]:
    """
    Cluster similar queries together.
    Returns list of query clusters.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Group these queries into semantic clusters (queries with similar intent/topic).
    
    Queries:
    {json.dumps(queries, indent=2)}
    
    Rules:
    - Each query should be in exactly one cluster
    - Clusters should represent distinct topics/intents
    - Name each cluster descriptively
    
    Output as JSON:
    ```json
    {{
        "clusters": [
            {{
                "name": "cluster name",
                "queries": ["query1", "query2"],
                "common_intent": "what these queries share"
            }}
        ]
    }}
    ```
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            result = json.loads(json_match.group())
            return result.get('clusters', [])
    except:
        pass
    
    return [{"name": "All Queries", "queries": queries, "common_intent": "Various"}]


def generate_content_brief(
    query: str,
    synthetic_queries: List[Dict],
    target_word_count: int = 1500
) -> Dict:
    """
    Generate a comprehensive content brief based on fan-out queries.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    queries_text = json.dumps(synthetic_queries, indent=2)
    
    prompt = f"""
    Create a comprehensive content brief for an article targeting this query cluster.
    
    ## PRIMARY QUERY
    {query}
    
    ## SYNTHETIC QUERIES TO COVER
    {queries_text}
    
    ## TARGET WORD COUNT
    {target_word_count} words
    
    Generate a content brief including:
    
    ```json
    {{
        "title_options": ["option 1", "option 2", "option 3"],
        "meta_description": "...",
        "target_audience": "...",
        "search_intent": "informational|transactional|navigational|comparative",
        "outline": [
            {{
                "section": "Introduction",
                "word_count": 150,
                "key_points": ["..."],
                "queries_covered": ["..."]
            }},
            {{
                "section": "H2 heading",
                "word_count": 300,
                "key_points": ["..."],
                "queries_covered": ["..."],
                "subsections": [
                    {{"heading": "H3", "points": ["..."]}}
                ]
            }}
        ],
        "faq_section": [
            {{"question": "...", "answer_length": 100, "covers_query": "..."}}
        ],
        "required_entities": ["entity1", "entity2"],
        "required_statistics": ["stat1", "stat2"],
        "internal_link_opportunities": ["topic1", "topic2"],
        "image_suggestions": [
            {{"type": "infographic|photo|screenshot", "description": "..."}}
        ],
        "cta_recommendation": "..."
    }}
    ```
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=4096
            )
        )
        
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        return {"error": str(e)}
    
    return {}


def batch_fanout_analysis(
    queries: List[str],
    api_key: str,
    language: str = "English",
    market: str = "Malaysia/Singapore"
) -> List[Dict]:
    """
    Run fan-out analysis on multiple queries at once.
    More efficient than running individually.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    
    prompt = BATCH_FANOUT_PROMPT.format(
        queries=queries_text,
        language=language,
        market=market
    )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192
            )
        )
        
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            result = json.loads(json_match.group())
            return result.get('results', [])
    except Exception as e:
        return [{"error": str(e)}]
    
    return []


def generate_optimization_recommendations(
    query: str,
    gaps: List[str],
    content_summary: str,
    api_key: str
) -> Dict:
    """
    Generate specific content optimization recommendations.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = CONTENT_OPTIMIZATION_PROMPT.format(
        query=query,
        gaps=json.dumps(gaps, indent=2),
        content_summary=content_summary[:3000]
    )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.5,
                max_output_tokens=4096
            )
        )
        
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        return {"error": str(e)}
    
    return {}


def compare_with_competitor(
    query: str,
    synthetic_queries: List[Dict],
    your_content: str,
    competitor_content: str,
    api_key: str
) -> Dict:
    """
    Compare your content with competitor against synthetic queries.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = COMPETITOR_ANALYSIS_PROMPT.format(
        query=query,
        synthetic_queries=json.dumps(synthetic_queries, indent=2),
        competitor_content=competitor_content[:5000],
        your_content=your_content[:5000]
    )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.5,
                max_output_tokens=4096
            )
        )
        
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        return {"error": str(e)}
    
    return {}


def optimize_passage(
    query: str,
    passage: str,
    api_key: str
) -> Dict:
    """
    Optimize a specific passage for better AI Mode retrieval.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = PASSAGE_OPTIMIZATION_PROMPT.format(
        query=query,
        passage=passage
    )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.5,
                max_output_tokens=2048
            )
        )
        
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        return {"error": str(e)}
    
    return {}


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_to_google_sheets_format(result: Dict) -> List[List]:
    """
    Convert results to format ready for Google Sheets import.
    """
    headers = ["Query", "Type", "Intent", "Priority", "Reasoning"]
    rows = [headers]
    
    for q in result.get('queries', []):
        rows.append([
            q.get('query', ''),
            q.get('type', ''),
            q.get('intent', ''),
            q.get('priority', ''),
            q.get('reasoning', '')
        ])
    
    return rows


def generate_seo_brief_markdown(
    query: str,
    fanout_result: Dict,
    content_brief: Dict
) -> str:
    """
    Generate a complete SEO brief in Markdown format.
    """
    md = f"""# SEO Content Brief

## Target Query
**Primary:** {query}

## Synthetic Queries to Cover
{chr(10).join([f"- {q.get('query', '')} ({q.get('type', '')})" for q in fanout_result.get('queries', [])[:15]])}

## Content Structure

### Title Options
{chr(10).join([f"- {t}" for t in content_brief.get('title_options', [])])}

### Meta Description
{content_brief.get('meta_description', 'N/A')}

### Target Audience
{content_brief.get('target_audience', 'N/A')}

### Search Intent
{content_brief.get('search_intent', 'N/A')}

## Content Outline

"""
    
    for section in content_brief.get('outline', []):
        md += f"### {section.get('section', 'Section')}\n"
        md += f"*Word count: ~{section.get('word_count', 0)} words*\n\n"
        md += "**Key Points:**\n"
        for point in section.get('key_points', []):
            md += f"- {point}\n"
        md += "\n**Queries Covered:**\n"
        for q in section.get('queries_covered', []):
            md += f"- {q}\n"
        md += "\n"
    
    md += """## FAQ Section

"""
    for faq in content_brief.get('faq_section', []):
        md += f"**Q: {faq.get('question', '')}**\n"
        md += f"*Answer length: ~{faq.get('answer_length', 100)} words*\n"
        md += f"*Covers: {faq.get('covers_query', '')}*\n\n"
    
    md += """## Required Elements

### Entities to Mention
"""
    for entity in content_brief.get('required_entities', []):
        md += f"- {entity}\n"
    
    md += """
### Statistics/Data Points
"""
    for stat in content_brief.get('required_statistics', []):
        md += f"- {stat}\n"
    
    md += """
### Internal Links
"""
    for link in content_brief.get('internal_link_opportunities', []):
        md += f"- Link to: {link}\n"
    
    md += f"""
## Call to Action
{content_brief.get('cta_recommendation', 'N/A')}

---
*Generated by YEA Query Fan-Out Tool*
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return md


# ============================================================================
# LANGUAGE-SPECIFIC UTILITIES
# ============================================================================

MARKET_CONTEXTS = {
    "Malaysia/Singapore": {
        "local_entities": ["Kuala Lumpur", "Singapore", "Johor Bahru", "Penang", "Selangor"],
        "languages": ["English", "Malay", "Chinese"],
        "local_platforms": ["Lazada", "Shopee", "Grab", "TikTok"],
        "search_engines": ["Google", "Baidu (for Chinese speakers)"]
    },
    "Southeast Asia": {
        "local_entities": ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam", "Philippines"],
        "languages": ["English", "Local languages"],
        "local_platforms": ["Lazada", "Shopee", "Tokopedia", "Grab", "Gojek"],
        "search_engines": ["Google"]
    },
    "Global": {
        "local_entities": [],
        "languages": ["English"],
        "local_platforms": ["Amazon", "Google", "Facebook"],
        "search_engines": ["Google", "Bing"]
    }
}


def get_market_context(market: str) -> Dict:
    """Get market-specific context for query generation."""
    return MARKET_CONTEXTS.get(market, MARKET_CONTEXTS["Global"])


def localize_queries(queries: List[Dict], market: str) -> List[Dict]:
    """Add market-specific variations to queries."""
    context = get_market_context(market)
    localized = queries.copy()
    
    # This would be expanded with actual localization logic
    return localized


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YEA Query Fan-Out Tool CLI")
    parser.add_argument("query", help="The query to analyze")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--mode", choices=["ai_mode", "ai_overview"], default="ai_mode")
    parser.add_argument("--language", default="English")
    parser.add_argument("--market", default="Malaysia/Singapore")
    parser.add_argument("--output", default="fanout_result.json")
    
    args = parser.parse_args()
    
    # Import main function from app
    from app import generate_fanout_queries, init_gemini
    
    if init_gemini(args.api_key):
        result = generate_fanout_queries(
            query=args.query,
            mode=args.mode,
            language=args.language,
            market=args.market
        )
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {args.output}")
        print(f"Generated {result.get('total_count', 0)} queries")
