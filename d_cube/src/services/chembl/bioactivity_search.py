"""
ChEMBL bioactivity search tool for the SMILES agent
"""
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .utils import ChEMBLClient

logger = logging.getLogger(__name__)

class ChEMBLBioactivityInput(BaseModel):
    """Input for ChEMBL bioactivity search"""
    target_query: str = Field(description="Target search query - can be target name, gene symbol, or ChEMBL target ID")
    compound_query: Optional[str] = Field(default=None, description="Optional compound query - ChEMBL ID or compound name")
    activity_type: Optional[str] = Field(default=None, description="Activity type filter (e.g., IC50, Ki, EC50)")
    assay_type: Optional[str] = Field(default=None, description="Assay type filter: A (ADME), B (Binding), F (Functional)")
    max_results: int = Field(default=20, description="Maximum number of results to return")
    include_inactive: bool = Field(default=False, description="Include activities without pChEMBL values")

class ChEMBLBioactivityTool(BaseTool):
    """Tool for searching bioactivity data in ChEMBL database"""
    
    name: str = "chembl_bioactivity_search"
    description: str = """
    Search for bioactivity data in the ChEMBL database. Can search by:
    - Target name, gene symbol, or ChEMBL target ID
    - Optionally filter by compound
    - Activity type (IC50, Ki, EC50, etc.)
    - Assay type (ADME, Binding, Functional)
    
    Returns bioactivity measurements including target information, compound details,
    activity values, and assay descriptions. Useful for drug discovery, target analysis,
    and understanding compound-target interactions.
    """
    args_schema: type[BaseModel] = ChEMBLBioactivityInput
    client: ChEMBLClient = None
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'client', ChEMBLClient())
    
    def _run(self, 
             target_query: str,
             compound_query: Optional[str] = None,
             activity_type: Optional[str] = None,
             assay_type: Optional[str] = None,
             max_results: int = 20,
             include_inactive: bool = False) -> str:
        """Execute ChEMBL bioactivity search"""
        
        try:
            # First, find the target
            targets = self._find_target(target_query)
            if not targets:
                return json.dumps({
                    "success": False,
                    "error": f"No targets found for query: {target_query}"
                })
            
            # Use the first target found
            target = targets[0]
            target_chembl_id = target['target_chembl_id']
            
            # Build activity filters
            activity_filters = {'target_chembl_id': target_chembl_id}
            
            if activity_type:
                activity_filters['standard_type'] = activity_type
            if assay_type:
                activity_filters['assay_type'] = assay_type
            if not include_inactive:
                activity_filters['pchembl_value__isnull'] = False
            
            # Filter by compound if specified
            if compound_query:
                compound_id = self._resolve_compound_id(compound_query)
                if compound_id:
                    activity_filters['molecule_chembl_id'] = compound_id
            
            # Get activities with limit to prevent fetching thousands
            activities = list(self.client.activity.filter(**activity_filters).only([
                'activity_id', 'molecule_chembl_id', 'target_chembl_id', 'assay_chembl_id',
                'standard_type', 'standard_value', 'standard_units', 'pchembl_value',
                'activity_comment', 'assay_description', 'assay_type', 'confidence_score',
                'document_chembl_id'
            ])[:max_results])  # Limit at query level
            
            if not activities:
                return json.dumps({
                    "success": True,
                    "target": target,
                    "message": f"No activities found for target {target_chembl_id} with given filters",
                    "activities_count": 0,
                    "activities": []
                })
            
            # Process activities for output
            processed_activities = []
            unique_molecule_ids = set()
            
            for activity in activities:
                processed_activity = {
                    "activity_id": activity.get("activity_id"),
                    "molecule_chembl_id": activity.get("molecule_chembl_id"),
                    "target_chembl_id": activity.get("target_chembl_id"),
                    "assay_chembl_id": activity.get("assay_chembl_id"),
                    "standard_type": activity.get("standard_type"),
                    "standard_value": activity.get("standard_value"),
                    "standard_units": activity.get("standard_units"),
                    "pchembl_value": activity.get("pchembl_value"),
                    "activity_comment": activity.get("activity_comment"),
                    "assay_description": activity.get("assay_description"),
                    "assay_type": activity.get("assay_type"),
                    "confidence_score": activity.get("confidence_score"),
                    "document_chembl_id": activity.get("document_chembl_id")
                }
                processed_activities.append(processed_activity)
                
                # Collect unique molecule IDs for batch name lookup
                if activity.get("molecule_chembl_id"):
                    unique_molecule_ids.add(activity.get("molecule_chembl_id"))
            
            # Batch fetch compound names instead of individual calls
            compound_names = {}
            if unique_molecule_ids:
                molecule_list = list(unique_molecule_ids)
                # Fetch molecules in batches to avoid API limits
                batch_size = 50
                for i in range(0, len(molecule_list), batch_size):
                    batch = molecule_list[i:i + batch_size]
                    try:
                        molecules = list(self.client.molecule.filter(
                            molecule_chembl_id__in=batch
                        ).only(['molecule_chembl_id', 'pref_name']))
                        
                        for mol in molecules:
                            if mol and mol.get('molecule_chembl_id') and mol.get('pref_name'):
                                compound_names[mol['molecule_chembl_id']] = mol['pref_name']
                    except Exception as batch_error:
                        logger.warning(f"Failed to fetch batch {i//batch_size + 1}: {batch_error}")
                        continue
            
            # Add compound names to activities
            for activity in processed_activities:
                mol_id = activity["molecule_chembl_id"]
                if mol_id in compound_names:
                    activity["compound_name"] = compound_names[mol_id]
            
            return json.dumps({
                "success": True,
                "target": {
                    "target_chembl_id": target["target_chembl_id"],
                    "pref_name": target["pref_name"],
                    "organism": target["organism"],
                    "target_type": target["target_type"]
                },
                "search_parameters": {
                    "target_query": target_query,
                    "compound_query": compound_query,
                    "activity_type": activity_type,
                    "assay_type": assay_type,
                    "include_inactive": include_inactive
                },
                "activities_count": len(processed_activities),
                "activities": processed_activities
            }, indent=2)
            
        except Exception as e:
            logger.error(f"ChEMBL bioactivity search error: {e}")
            return json.dumps({
                "success": False,
                "error": f"Bioactivity search failed: {str(e)}"
            })
    
    def _find_target(self, query: str) -> List[Dict[str, Any]]:
        """Find target by query string"""
        # First try exact ChEMBL ID match
        if query.startswith('CHEMBL'):
            try:
                target_data = list(self.client.target.filter(target_chembl_id=query))
                if target_data:
                    return target_data
            except:
                pass
        
        # Try gene name search
        return self.client.search_targets_by_gene(query)
    
    def _resolve_compound_id(self, compound_query: str) -> Optional[str]:
        """Resolve compound query to ChEMBL ID"""
        if compound_query.startswith('CHEMBL'):
            return compound_query
        
        # Search by name
        compounds = self.client.search_molecule_by_name(compound_query)
        return compounds[0]['molecule_chembl_id'] if compounds else None
    
    def _get_compound_names(self, activities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get compound names for molecules in activities"""
        compound_names = {}
        unique_mol_ids = set(a["molecule_chembl_id"] for a in activities if a["molecule_chembl_id"])
        
        for mol_id in unique_mol_ids:
            try:
                mol_data = self.client.get_molecule_by_chembl_id(mol_id, include_structures=False)
                if mol_data and mol_data.get("pref_name"):
                    compound_names[mol_id] = mol_data["pref_name"]
            except:
                continue
                
        return compound_names
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version of the tool"""
        return self._run(*args, **kwargs)