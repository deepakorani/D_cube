import asyncio
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from src.modules.agent import Agent

async def test_smiles_agent():
    """Test the SMILES agent with various queries"""
    # Initialize agent
    checkpointer = MemorySaver()
    agent = Agent(checkpointer)
    thread_id = {"configurable": {"thread_id": "test_session"}}
    
    # Test queries
    test_queries = [
        # "Load the CSV file test.csv and show me the first 5 rows",
        # "Load test.csv and then standardize all the SMILES in it",
        # "Load test.csv and then give me the descriptors for it", 
        "Find compounds similar to given SMILES query of CO[C@@H](CCC#C\C=C/CCCC(C)CCCCC=C)C(=O)[O-] with similarity threshold of 70%",
        # "What bioactivity data exists for BRD4 inhibitors?",
        "Find compounds with IC50 < 100nM against this target"
    ]
    
    print("🧪 Testing D_CUBE SMILES Agent\n" + "="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 60)
        
        try:
            # Stream the response
            async for chunk in agent.graph.astream(
                {"messages": [HumanMessage(content=query)]},
                config=thread_id
            ):
                # Handle assistant messages (including tool calls)
                if "assistant" in chunk:
                    for message in chunk["assistant"]["messages"]:
                        if isinstance(message, AIMessage):
                            # Show AI content
                            if hasattr(message, 'content') and message.content:
                                print(f"🤖 Assistant: {message.content}")
                            
                            # Show tool calls being made
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    print(f"🔧 Calling tool: {tool_call['name']}")
                                    print(f"   Arguments: {json.dumps(tool_call['args'], indent=2)}")
                
                # Handle tool results
                elif "tools" in chunk:
                    for message in chunk["tools"]["messages"]:
                        if isinstance(message, ToolMessage):
                            print(f"🔧 Tool '{message.name if hasattr(message, 'name') else 'Unknown'}' result:")
                            
                            # Try to parse JSON results for better formatting
                            try:
                                result = json.loads(message.content)
                                if isinstance(result, dict):
                                    # Pretty print structured results
                                    if 'success' in result:
                                        status = "✅" if result['success'] else "❌"
                                        print(f"   {status} Success: {result['success']}")
                                        
                                        if 'message' in result:
                                            print(f"   📋 Message: {result['message']}")
                                            
                                        if 'data' in result and result['data']:
                                            print(f"   📊 Data preview (first few rows):")
                                            data = result['data']
                                            if isinstance(data, list) and len(data) > 0:
                                                # Show first few rows of data
                                                for idx, row in enumerate(data[:5]):
                                                    print(f"      Row {idx+1}: {row}")
                                                if len(data) > 5:
                                                    print(f"      ... and {len(data)-5} more rows")
                                        
                                        if 'results' in result and result['results']:
                                            print(f"   🧪 Results preview:")
                                            results = result['results']
                                            if isinstance(results, list) and len(results) > 0:
                                                for idx, res in enumerate(results[:3]):
                                                    print(f"      Result {idx+1}: {res}")
                                                if len(results) > 3:
                                                    print(f"      ... and {len(results)-3} more results")
                                    else:
                                        # Handle other structured data
                                        print(f"   📋 Result: {json.dumps(result, indent=4)[:500]}...")
                                else:
                                    print(f"   📋 Raw result: {str(result)[:500]}...")
                                    
                            except (json.JSONDecodeError, TypeError):
                                # Handle non-JSON results
                                content = str(message.content)
                                if len(content) > 1000:
                                    print(f"   📋 Result (truncated): {content[:1000]}...")
                                else:
                                    print(f"   📋 Result: {content}")
                                    
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_smiles_agent())