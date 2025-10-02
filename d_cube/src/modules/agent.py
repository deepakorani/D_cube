from typing import TypedDict, Annotated, AsyncGenerator, Dict, List
import json
import logging

from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages

from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage

# Import SMILES tools
from src.services.smiles.descriptor_2d import Descriptors2DTool
from src.services.smiles.smiles_standardize import SmilesStandardizeTool
from src.services.data.csv_loader import CSVDataLoaderTool
from src.services.advanced_cheminformatics.protonation_tool import ProtonationTool
from src.services.advanced_cheminformatics.conformer_tool import ConformerTool
from src.services.advanced_cheminformatics.sdf_export_tool import SDFExportTool
from src.services.chembl import get_chembl_tools
from src.utils.langchain_setup import chat_model
from src.modules.prompts import SYSTEM_PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Agent:
    def __init__(self, checkpointer):
        self.system = SYSTEM_PROMPT
        self.logger = logger
        
        graph = StateGraph(AgentState)
        graph.add_node("assistant", self.acall_openai)
        tools = self.get_tools()
        graph.add_node("tools", self.create_tool_node_with_fallback(tools))
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")
        graph.set_entry_point("assistant")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = chat_model.bind_tools(tools)

    def get_tools(self):
        """Get tools for drug discovery: SMILES processing + data loading"""
        tools = [
            SmilesStandardizeTool(),
            Descriptors2DTool(),
            CSVDataLoaderTool(),
            *get_chembl_tools(),
             # Advanced cheminformatics tools
            ProtonationTool(),
            ConformerTool(),
            SDFExportTool()
        ]
        return tools

    def wrap_model(self) -> RunnableSerializable[AgentState, AIMessage]:
        model = self.model.bind_tools(self.get_tools())
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=self.system)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | model

    async def acall_openai(self, state: AgentState) -> AsyncGenerator[Dict[str, List[BaseMessage]], None]:
        model = self.wrap_model()
        try:
            # Uncomment the following to not stream token by token
            # response = await model.ainvoke(state)
            # return {"messages": [response]}
            first = True
            is_tool_call = False
            async for chunk in model.astream(state):
                if len(chunk.tool_call_chunks):
                    is_tool_call = True
                    if first:
                        gathered = chunk
                        first = False
                    else:
                        gathered = gathered + chunk
                else:
                    yield {"messages": [chunk]}
            if is_tool_call:
                yield {"messages": [gathered]}
        except Exception as e:
            # Handle any streaming errors
            self.logger.error(f"Error during streaming: {e}")
            yield {"messages": [AIMessage(content=f"Error: {str(e)}")]}

    def handle_tool_error(self, state: AgentState):
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def create_tool_node_with_fallback(self, tools: list):
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )
    
    # SMILES-specific helper methods
    async def process_smiles_standardization_result(self, tool_result: str) -> Dict[str, any]:
        """Process the result from SMILES standardization tool"""
        try:
            if isinstance(tool_result, str):
                result = json.loads(tool_result)
            else:
                result = tool_result
            
            if result.get('success'):
                return {
                    'status': 'success',
                    'message': f"Successfully processed {result.get('processed_count', 0)} SMILES",
                    'valid_count': result.get('valid_count', 0),
                    'success_rate': result.get('success_rate', 0),
                    'results': result.get('results')
                }
            else:
                return {
                    'status': 'error',
                    'message': result.get('error', 'Unknown error in SMILES standardization')
                }
        except Exception as e:
            self.logger.error(f"Error processing SMILES standardization result: {e}")
            return {
                'status': 'error',
                'message': f"Error processing result: {str(e)}"
            }

    async def process_descriptors_result(self, tool_result: str) -> Dict[str, any]:
        """Process the result from 2D descriptors calculation tool"""
        try:
            if isinstance(tool_result, str):
                result = json.loads(tool_result)
            else:
                result = tool_result
            
            if result.get('success'):
                return {
                    'status': 'success',
                    'message': f"Successfully calculated descriptors for {result.get('valid_count', 0)} molecules",
                    'processed_count': result.get('processed_count', 0),
                    'success_rate': result.get('success_rate', 0),
                    'results': result.get('results')
                }
            else:
                return {
                    'status': 'error',
                    'message': result.get('error', 'Unknown error in descriptor calculation')
                }
        except Exception as e:
            self.logger.error(f"Error processing descriptors result: {e}")
            return {
                'status': 'error',
                'message': f"Error processing result: {str(e)}"
            }