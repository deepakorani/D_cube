# src/services/chembl/compound_search.py
"""
ChEMBL compound search tool for the SMILES agent
"""
import json
import logging
from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .utils import ChEMBLClient

logger = logging.getLogger(__name__)

class ChEMBLCompoundSearchInput(BaseModel):
    """Input for ChEMBL compound search"""
    query: str = Field(description="Search query - can be compound name, ChEMBL ID, or InChI key")
    search_type: str = Field(
        default="name", 
        description="Type of search: 'name', 'chembl_id', 'inchi_key', 'similarity'"
    )
    similarity_threshold: int = Field(default=70, description="Similarity threshold for similarity search (0-100)")
    include_properties: bool = Field(default=True, description="Include molecular properties in results")
    include_structures: bool = Field(default=True, description="Include molecular structures in results")
    max_results: int = Field(default=10, description="Maximum number of results to return")

class ChEMBLCompoundSearchTool(BaseTool):
    """Tool for searching compounds in ChEMBL database"""
    
    name: str = "chembl_compound_search"
    description: str = """
    Search for compounds in the ChEMBL database. Can search by:
    - Compound name (preferred name or synonyms)
    - ChEMBL ID (e.g., CHEMBL25)
    - InChI key
    - Similarity to SMILES structure
    
    Returns compound information including ChEMBL ID, name, molecular properties, and structures.
    Useful for finding bioactivity data, drug information, and chemical structures.
    """
    args_schema: type[BaseModel] = ChEMBLCompoundSearchInput
    client: ChEMBLClient = None
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'client', ChEMBLClient())
    
    def _run(self, 
             query: str,
             search_type: str = "name",
             similarity_threshold: int = 70,
             include_properties: bool = True,
             include_structures: bool = True,
             max_results: int = 10) -> str:
        """Execute ChEMBL compound search"""
        
        try:
            results = []
            
            if search_type == "name":
                results = self.client.search_molecule_by_name(query, exact=False)
                
            elif search_type == "chembl_id":
                result = self.client.get_molecule_by_chembl_id(query, include_structures)
                results = [result] if result else []
                
            elif search_type == "inchi_key":
                results = self.client.get_molecules_by_inchi_key(query)
                
            elif search_type == "similarity":
                # Assume query is SMILES for similarity search
                similarity_results = self.client.find_similar_compounds(
                    smiles=query, similarity=similarity_threshold
                )
                
                # Get full molecule data for similar compounds
                for sim_result in similarity_results[:max_results]:
                    mol_data = self.client.get_molecule_by_chembl_id(
                        sim_result['molecule_chembl_id'], include_structures
                    )
                    if mol_data:
                        mol_data['similarity'] = sim_result['similarity']
                        results.append(mol_data)
                        
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown search type: {search_type}",
                    "valid_types": ["name", "chembl_id", "inchi_key", "similarity"]
                })
            
            # Limit results
            if len(results) > max_results:
                results = results[:max_results]
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = {
                    "molecule_chembl_id": result.get("molecule_chembl_id"),
                    "pref_name": result.get("pref_name"),
                }
                
                # Add similarity if present
                if "similarity" in result:
                    processed_result["similarity"] = result["similarity"]
                
                # Add properties if requested
                if include_properties and "molecule_properties" in result:
                    processed_result["properties"] = result["molecule_properties"]
                
                # Add structures if requested
                if include_structures and "molecule_structures" in result:
                    structures = result["molecule_structures"]
                    processed_result["structures"] = {
                        "smiles": structures.get("canonical_smiles"),
                        "inchi": structures.get("standard_inchi"),
                        "inchi_key": structures.get("standard_inchi_key")
                    }
                
                # Add other relevant fields
                if "max_phase" in result:
                    processed_result["max_phase"] = result["max_phase"]
                if "indication_class" in result:
                    processed_result["indication_class"] = result["indication_class"]
                    
                processed_results.append(processed_result)
            
            return json.dumps({
                "success": True,
                "search_type": search_type,
                "query": query,
                "results_count": len(processed_results),
                "results": processed_results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"ChEMBL compound search error: {e}")
            return json.dumps({
                "success": False,
                "error": f"Search failed: {str(e)}"
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version of the tool"""
        return self._run(*args, **kwargs)