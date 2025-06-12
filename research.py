import os
import asyncio
import streamlit as st
from typing import List, Dict, Any
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import ollama

# Local Model Wrapper
class LocalModel:
    def __init__(self, model_name="llama2"):  # Changed default to llama2
        self.model_name = model_name
    
    async def generate(self, prompt: str, max_length: int = 300, **kwargs) -> str:
        response = await ollama.AsyncClient().generate(
            model=self.model_name,
            prompt=prompt,
            options={
                'num_predict': max_length,
                'temperature': kwargs.get('temperature', 0.7)
            }
        )
        return response['response']

# Initialize models
@st.cache_resource
def load_models():
    return {
        'research': LocalModel("llama2"),  # Using llama2 for research
        'editor': LocalModel("mistral")    # Using mistral for better quality generation
    }

# Data Models
class ResearchPlan(BaseModel):
    topic: str
    search_queries: List[str]
    focus_areas: List[str]

class ResearchReport(BaseModel):
    title: str
    outline: List[str]
    report: str
    sources: List[str]
    word_count: int = 0

async def generate_research_plan(topic: str, model) -> ResearchPlan:
    prompt = f"""Create a research plan for: {topic}
    
    Return the response in this exact format:
    Topic: [clear topic statement]
    Search Queries:
    1. [query 1]
    2. [query 2]
    3. [query 3]
    Focus Areas:
    - [focus 1]
    - [focus 2]
    - [focus 3]"""
    
    response = await model.generate(prompt, temperature=0.7)
    
    # Parse the response
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    topic_line = next((line for line in lines if line.lower().startswith('topic:')), f"Topic: {topic}")
    queries = [line.split(' ', 1)[1] for line in lines if line.lower().startswith(('1.', '2.', '3.'))]
    focuses = [line[2:].strip() for line in lines if line.startswith('-')]
    
    return ResearchPlan(
        topic=topic_line.split(':', 1)[1].strip() if ':' in topic_line else topic,
        search_queries=queries[:3] if queries else [f"information about {topic}"],
        focus_areas=focuses[:3] if focuses else [topic]
    )

# Research Functions
async def research_topic(plan: ResearchPlan, model) -> List[Dict[str, str]]:
    """Simulate research using the LLM instead of web search"""
    research_results = []
    for query in plan.search_queries:
        prompt = f"""Based on your training data, provide information about: {query}
        Format your response as:
        Title: [Short title]
        Summary: [2-3 sentence summary]
        Key Points:
        - Point 1
        - Point 2
        - Point 3"""
        
        response = await model.generate(prompt)
        research_results.append({
            'url': f"llm_knowledge:{query.replace(' ', '_')}",
            'title': query,
            'snippet': response
        })
    return research_results

async def generate_report(topic: str, research_data: List[Dict[str, str]], model) -> ResearchReport:
    context = "\n".join([f"Source: {r['url']}\nContent: {r['snippet']}" for r in research_data])
    
    prompt = f"""Create a comprehensive report about: {topic}
    
    Research Context:
    {context}
    
    Format your response as a markdown document with these sections:
    # [Title]
    
    ## Outline
    1. [Section 1]
    2. [Section 2]
    ...
    
    ## Report
    [Full report content here...]
    
    ## Sources
    - [Source 1]
    - [Source 2]
    ..."""
    
    response = await model.generate(prompt, max_length=1500, temperature=0.7)
    
    return ResearchReport(
        title=topic,
        outline=[],  # You might want to parse this from the response
        report=response,
        sources=[r['url'] for r in research_data],
        word_count=len(response.split()) if isinstance(response, str) else 0
    )

# Streamlit UI
st.set_page_config(
    page_title="Local Research Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📰 Local Research Agent")
st.markdown("""
This app demonstrates a research agent using local language models.
""")

# Sidebar
with st.sidebar:
    st.header("Research Topic")
    user_topic = st.text_input("Enter a topic to research:")
    start_button = st.button("Start Research", disabled=not user_topic)

# Main content
tab1, tab2 = st.tabs(["Research Process", "Report"])

# Session state
if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "report" not in st.session_state:
    st.session_state.report = None

async def run_research(topic: str):
    models = load_models()
    
    with st.spinner("Creating research plan..."):
        plan = await generate_research_plan(topic, models['research'])
        
        with tab1:
            st.subheader("Research Plan")
            st.json(plan.dict())
            
            st.subheader("Gathering Information...")
            research_data = await research_topic(plan, models['research'])
            st.session_state.research_data = research_data
            
            st.subheader("Generating Report...")
            report = await generate_report(topic, research_data, models['editor'])
            st.session_state.report = report
            st.success("Research Complete!")

# Run research
if start_button and user_topic:
    asyncio.run(run_research(user_topic))

# Display results
if st.session_state.report:
    with tab2:
        report = st.session_state.report
        st.markdown(f"# {report.title}")
        
        st.markdown("## Report")
        st.markdown(report.report)
        
        with st.expander("Sources"):
            for i, source in enumerate(report.sources, 1):
                st.markdown(f"{i}. {source}")

        st.download_button(
            label="Download Report",
            data=report.report,
            file_name=f"{report.title.replace(' ', '_')}.md",
            mime="text/markdown"
        )