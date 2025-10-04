# simple_agent_test.py - Simple Streamlit frontend for testing the SMILES agent
import streamlit as st
import asyncio
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from src.modules.agent import Agent

# Page configuration
st.set_page_config(
    page_title="D_CUBE SMILES Agent Test",
    page_icon="🧪",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = "test_session"
if 'messages' not in st.session_state:
    st.session_state.messages = []

def initialize_agent():
    """Initialize the agent"""
    try:
        checkpointer = MemorySaver()
        agent = Agent(checkpointer)
        return agent, None
    except Exception as e:
        return None, str(e)

def run_agent_query(agent, query, session_id):
    """Run a query through the agent"""
    try:
        config = {"configurable": {"thread_id": session_id}}
        
        # Collect results
        assistant_messages = []
        tool_results = []
        
        # Run the agent stream
        async def run_stream():
            async for chunk in agent.graph.astream(
                {"messages": [HumanMessage(content=query)]},
                config=config
            ):
                if "assistant" in chunk:
                    for message in chunk["assistant"]["messages"]:
                        if hasattr(message, 'content') and message.content:
                            assistant_messages.append(message.content)
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                tool_results.append({
                                    "type": "tool_call",
                                    "tool": tool_call['name'],
                                    "args": tool_call['args']
                                })
                
                elif "tools" in chunk:
                    for message in chunk["tools"]["messages"]:
                        if hasattr(message, 'content'):
                            try:
                                result = json.loads(message.content)
                                tool_results.append({
                                    "type": "tool_result",
                                    "content": result
                                })
                            except:
                                tool_results.append({
                                    "type": "tool_result",
                                    "content": str(message.content)
                                })
        
        # Run the async function
        asyncio.run(run_stream())
        
        return {
            "success": True,
            "assistant_messages": assistant_messages,
            "tool_results": tool_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def display_tool_result(result):
    """Display tool results in a formatted way"""
    if isinstance(result, dict):
        if result.get('success'):
            st.success("✅ Tool executed successfully")
            
            # Display key metrics if available
            if 'results_count' in result:
                st.metric("Results Found", result['results_count'])
            if 'activities_count' in result:
                st.metric("Activities Found", result['activities_count'])
            if 'total_results' in result:
                st.metric("Total Results", result['total_results'])
            
            if 'processed_count' in result:
                col1, col2, col3 = st.columns(3)
                col1.metric("Processed", result.get('processed_count', 0))
                col2.metric("Valid", result.get('valid_count', 0))
                if 'success_rate' in result:
                    col3.metric("Success Rate", f"{result['success_rate']:.1%}")
            
            # Display target information FIRST
            if 'target' in result and result['target']:
                st.subheader("Target Information")
                target_info = result['target']
                col1, col2 = st.columns(2)
                col1.metric("Target ID", target_info.get('target_chembl_id', 'N/A'))
                col2.metric("Target Name", target_info.get('pref_name', 'N/A'))
                if target_info.get('organism'):
                    st.info(f"Organism: {target_info['organism']}")
                if target_info.get('target_type'):
                    st.info(f"Type: {target_info['target_type']}")
            
            # Display activities (for bioactivity results) - CHECK THIS FIRST
            if 'activities' in result:
                st.subheader("Bioactivity Data")
                activities = result['activities']
                
                if activities and isinstance(activities, list) and len(activities) > 0:
                    try:
                        # Convert to DataFrame
                        activities_df = pd.DataFrame(activities)
                        
                        # Show key columns first
                        display_columns = ['compound_name', 'molecule_chembl_id', 'standard_type', 
                                         'standard_value', 'standard_units', 'pchembl_value']
                        
                        # Only show columns that exist
                        available_columns = [col for col in display_columns if col in activities_df.columns]
                        
                        if available_columns:
                            st.dataframe(activities_df[available_columns], width="stretch")
                        else:
                            st.dataframe(activities_df, width="stretch")
                            
                        # Show full data in expander
                        with st.expander("Full Activity Data"):
                            st.dataframe(activities_df, width="stretch")
                            
                    except Exception as e:
                        st.error(f"Error displaying activities as DataFrame: {e}")
                        st.json(activities)
                else:
                    st.info("No activities found or activities data is empty")
            
            # Display ChEMBL simple search results
            if 'results' in result and result['results']:
                # Handle nested ChEMBL results structure
                if isinstance(result['results'], dict):
                    # Simple search results with targets and compounds
                    if 'targets' in result['results'] and result['results']['targets']:
                        st.subheader("Targets Found")
                        targets_df = pd.DataFrame(result['results']['targets'])
                        st.dataframe(targets_df, width="stretch")
                    
                    if 'compounds' in result['results'] and result['results']['compounds']:
                        st.subheader("Compounds Found")
                        compounds_df = pd.DataFrame(result['results']['compounds'])
                        st.dataframe(compounds_df, width="stretch")
                        
                elif isinstance(result['results'], list) and len(result['results']) > 0:
                    st.subheader("Results")
                    try:
                        df = pd.DataFrame(result['results'])
                        st.dataframe(df, width="stretch")
                    except:
                        st.json(result['results'])
                else:
                    st.json(result['results'])
            
            # Display general data tables (for SMILES tools)
            elif 'data' in result and result['data']:
                st.subheader("Data")
                if isinstance(result['data'], list):
                    try:
                        df = pd.DataFrame(result['data'])
                        st.dataframe(df, width="stretch")
                    except:
                        st.json(result['data'])
                else:
                    st.json(result['data'])
            
            # Display search parameters
            if 'search_parameters' in result:
                with st.expander("Search Parameters"):
                    st.json(result['search_parameters'])
            
            # Display query information
            if 'query' in result:
                st.info(f"Query: {result['query']}")
            if 'search_type' in result:
                st.info(f"Search Type: {result['search_type']}")
                
        else:
            st.error(f"❌ Tool failed: {result.get('error', 'Unknown error')}")
            
        # Always show raw result for debugging
        with st.expander("Raw Result Data (Debug)"):
            st.json(result)
    else:
        st.text(str(result))

# Main UI
st.title("🧪 D_CUBE SMILES Agent Test")
st.markdown("Test your SMILES agent with ChEMBL integration")

# Initialize agent
if st.session_state.agent is None:
    with st.spinner("Initializing agent..."):
        agent, error = initialize_agent()
        if agent:
            st.session_state.agent = agent
            st.success("✅ Agent initialized successfully!")
        else:
            st.error(f"❌ Failed to initialize agent: {error}")
            st.stop()

# Test queries section
st.header("Quick Test Queries")

# Quick test buttons
st.subheader("Quick Tests")
cols = st.columns(3)

# Optimized test queries
optimized_test_queries = {
    "SMILES Standardization": "Standardize this SMILES: CCO",
    "Descriptor Calculation": "Calculate Lipinski descriptors for aspirin: CC(=O)Oc1ccccc1C(=O)O",
    "Quick Target Search": "Use simple search to find BRD4 target in ChEMBL",
    "Quick Compound Search": "Use simple search to find aspirin in ChEMBL", 
    "hERG IC50 Data": "Find IC50 bioactivity data for hERG target (limit 10 results)",
    "Drug Discovery": "Find approved drugs for diabetes (max 5 results)",
    "CSV Processing": "Load test_data.csv and standardize all SMILES",
}

for i, (name, query) in enumerate(optimized_test_queries.items()):
    col = cols[i % 3]
    if col.button(name, key=f"test_{i}"):
        st.session_state.current_query = query

# Custom query input
st.subheader("Custom Query")
custom_query = st.text_area(
    "Enter your query:",
    height=100,
    placeholder="Ask me to standardize SMILES, search ChEMBL, calculate descriptors, or any other chemistry question!"
)

col1, col2 = st.columns([1, 4])
with col1:
    run_query = st.button("Run Query", type="primary")
with col2:
    clear_history = st.button("Clear History")

# Use current query from quick tests or custom input
query_to_run = None
if hasattr(st.session_state, 'current_query'):
    query_to_run = st.session_state.current_query
    delattr(st.session_state, 'current_query')
elif run_query and custom_query.strip():
    query_to_run = custom_query.strip()

# Clear history
if clear_history:
    st.session_state.messages = []
    st.rerun()

# Run query
if query_to_run:
    st.session_state.messages.append({"type": "user", "content": query_to_run})
    
    with st.spinner("Processing query..."):
        result = run_agent_query(
            st.session_state.agent, 
            query_to_run, 
            st.session_state.session_id
        )
    
    st.session_state.messages.append({"type": "assistant", "content": result})
    st.rerun()

# Display conversation history
if st.session_state.messages:
    st.header("Conversation History")
    
    for i, message in enumerate(st.session_state.messages):
        if message["type"] == "user":
            st.markdown(f"**🤔 You:** {message['content']}")
        
        elif message["type"] == "assistant":
            result = message["content"]
            
            if result["success"]:
                st.markdown("**🤖 Assistant:**")
                
                # Display assistant messages
                if result["assistant_messages"]:
                    for msg in result["assistant_messages"]:
                        st.markdown(f"💬 {msg}")
                
                # Display tool results
                if result["tool_results"]:
                    st.markdown("**🔧 Tool Results:**")
                    for j, tool_result in enumerate(result["tool_results"]):
                        if tool_result["type"] == "tool_call":
                            with st.expander(f"🔧 {tool_result['tool']} - Call {j+1}"):
                                st.json(tool_result["args"])
                        
                        elif tool_result["type"] == "tool_result":
                            with st.expander(f"📊 Tool Result {j+1}", expanded=True):
                                display_tool_result(tool_result["content"])
            else:
                st.error(f"❌ Error: {result['error']}")
        
        st.divider()

# Sidebar with agent info
st.sidebar.title("Agent Information")
st.sidebar.info(f"Session ID: {st.session_state.session_id}")

# Available tools info
if st.session_state.agent:
    st.sidebar.subheader("Available Tools")
    tools = st.session_state.agent.tools
    for tool_name, tool in tools.items():
        st.sidebar.text(f"• {tool_name}")

# Instructions
st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
**Available Operations:**
- SMILES standardization
- Descriptor calculations
- ChEMBL compound searches
- ChEMBL bioactivity data
- ChEMBL drug discovery
- CSV file processing
- Similarity searches
- **Protonation state calculation**
- **Conformer generation**
- **SDF file export**

**Example Queries:**
- "Standardize CCO"
- "Find aspirin in ChEMBL"
- "Calculate descriptors for benzene"
- "Get IC50 data for hERG"
- "Calculate protonation states for aspirin at pH 7.4"
- "Generate 10 conformers for ibuprofen"
- "Export conformers to SDF files"

**Advanced Workflow:**
- "Load CSV, filter compounds with MW<500, calculate protonation states, generate conformers, export SDF files"
""")

# Add new advanced section
st.sidebar.subheader("Advanced Cheminformatics")
st.sidebar.markdown("""
**New Capabilities:**
- **Protonation:** pH-dependent ionization states
- **Conformers:** 3D structure generation
- **SDF Export:** Multi-conformer files for modeling

**Workflow Example:**
1. Load molecules from CSV
2. Filter by drug-like properties
3. Calculate protonation at pH 7.4
4. Generate multiple conformers
5. Export individual SDF files
""")

# Add a file uploader for CSV files
st.sidebar.subheader("File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type="csv",
    help="Upload a CSV file containing SMILES data"
)

if uploaded_file is not None:
    st.sidebar.success(f"File uploaded: {uploaded_file.name}")
    if st.sidebar.button("Process Uploaded File"):
        st.session_state.current_query = f"Load and process the uploaded CSV file {uploaded_file.name}"

# Debug info
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("Debug Information")
    st.sidebar.json({
        "session_id": st.session_state.session_id,
        "messages_count": len(st.session_state.messages),
        "agent_initialized": st.session_state.agent is not None
    })