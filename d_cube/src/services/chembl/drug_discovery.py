# src/services/chembl/drug_discovery.py
"""
ChEMBL drug discovery tool for the SMILES agent
"""
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .utils import ChEMBLClient

logger = logging.getLogger(__name__)

class ChEMBLDrugDiscoveryInput(BaseModel):
    """Input for ChEMBL drug discovery search"""
    query_type: str = Field(
        description="Type of drug discovery query: 'approved_drugs', 'clinical_candidates', 'disease_drugs', 'target_drugs', 'similar_drugs'"
    )
    query_value: Optional[str] = Field(default=None, description="Query value (disease name, target name, compound for similarity, etc.)")
    phase_filter: Optional[int] = Field(default=None, description="Development phase filter (0-4, 4=approved)")
    approval_year: Optional[int] = Field(default=None, description="Minimum approval year for approved drugs")
    similarity_threshold: int = Field(default=70, description="Similarity threshold for similar_drugs query")
    max_results: int = Field(default=20, description="Maximum number of results to return")
    include_properties: bool = Field(default=True, description="Include molecular properties")

class ChEMBLDrugDiscoveryTool(BaseTool):
    """Tool for drug discovery queries in ChEMBL database"""
    
    name: str = "chembl_drug_discovery"
    description: str = """
    Perform drug discovery-focused searches in ChEMBL database:
    - approved_drugs: Find approved drugs (phase 4) with optional filters
    - clinical_candidates: Find compounds in clinical trials (phases 1-3)
    - disease_drugs: Find drugs for specific diseases/indications
    - target_drugs: Find drugs targeting specific proteins/genes
    - similar_drugs: Find drugs similar to a given compound
    
    Returns drug information including development phase, approval status, 
    indications, molecular properties, and target information.
    """
    args_schema: type[BaseModel] = ChEMBLDrugDiscoveryInput
    client: ChEMBLClient = None
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'client', ChEMBLClient())
    
    def _run(self, 
             query_type: str,
             query_value: Optional[str] = None,
             phase_filter: Optional[int] = None,
             approval_year: Optional[int] = None,
             similarity_threshold: int = 70,
             max_results: int = 20,
             include_properties: bool = True) -> str:
        """Execute ChEMBL drug discovery search"""
        
        try:
            results = []
            
            if query_type == "approved_drugs":
                if query_value:
                    # Search for approved drugs for specific disease
                    results = self._get_approved_drugs_by_disease(query_value, max_results)
                else:
                    # Get general approved drugs
                    results = self._get_approved_drugs(approval_year, max_results)
                
            elif query_type == "clinical_candidates":
                results = self._get_clinical_candidates(phase_filter, max_results)
                
            elif query_type == "disease_drugs":
                if not query_value:
                    return json.dumps({
                        "success": False,
                        "error": "query_value required for disease_drugs search"
                    })
                results = self._get_disease_drugs(query_value, max_results)
                
            elif query_type == "target_drugs":
                if not query_value:
                    return json.dumps({
                        "success": False,
                        "error": "query_value required for target_drugs search"
                    })
                results = self._get_target_drugs(query_value, max_results)
                
            elif query_type == "similar_drugs":
                if not query_value:
                    return json.dumps({
                        "success": False,
                        "error": "query_value (SMILES or ChEMBL ID) required for similar_drugs search"
                    })
                results = self._get_similar_drugs(query_value, similarity_threshold, max_results)
                
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown query_type: {query_type}",
                    "valid_types": ["approved_drugs", "clinical_candidates", "disease_drugs", "target_drugs", "similar_drugs"]
                })
            
            # Process results
            processed_results = self._process_drug_results(results, include_properties)
            
            return json.dumps({
                "success": True,
                "query_type": query_type,
                "query_value": query_value,
                "results_count": len(processed_results),
                "results": processed_results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"ChEMBL drug discovery error: {e}")
            return json.dumps({
                "success": False,
                "error": f"Drug discovery search failed: {str(e)}"
            })
    
    def _get_approved_drugs(self, approval_year: Optional[int], max_results: int) -> List[Dict[str, Any]]:
        """Get approved drugs using efficient filtering"""
        try:
            filters = {'max_phase': 4}  # Approved drugs
            
            if approval_year:
                filters['first_approval__gte'] = approval_year
            
            # Get approved drugs with essential fields only - much faster
            approved_drugs = list(self.client.molecule.filter(**filters).only([
                'molecule_chembl_id', 'pref_name', 'max_phase', 'first_approval',
                'indication_class', 'therapeutic_flag', 'oral', 'parenteral', 'topical'
            ])[:max_results])  # Limit at query level
            
            return approved_drugs
            
        except Exception as e:
            logger.error(f"Error in _get_approved_drugs: {e}")
            return []
    
    def _get_approved_drugs_by_disease(self, disease_term: str, max_results: int) -> List[Dict[str, Any]]:
        """Get approved drugs for a specific disease - optimized version"""
        try:
            # Method 1: Direct search on approved drugs with indication filter
            approved_disease_drugs = list(self.client.molecule.filter(
                max_phase=4,  # Approved drugs only
                indication_class__icontains=disease_term
            ).only(['molecule_chembl_id', 'pref_name', 'max_phase', 'indication_class', 'first_approval'])[:max_results])
            
            if approved_disease_drugs:
                return approved_disease_drugs
            
            # Method 2: Use drug_indication approach for approved drugs only
            try:
                # Get indications for the disease
                indications = list(self.client.drug_indication.filter(
                    efo_term__icontains=disease_term,
                    max_phase_for_ind=4  # Only approved indications
                ).only(['molecule_chembl_id'])[:max_results])
                
                if indications:
                    molecule_ids = [
                        ind['molecule_chembl_id'] for ind in indications 
                        if ind.get('molecule_chembl_id')
                    ]
                    unique_molecule_ids = list(set(molecule_ids))[:max_results]
                    
                    if unique_molecule_ids:
                        # Get only approved molecules
                        approved_drugs = list(self.client.molecule.filter(
                            molecule_chembl_id__in=unique_molecule_ids,
                            max_phase=4  # Ensure they're approved
                        ).only(['molecule_chembl_id', 'pref_name', 'max_phase', 'indication_class', 'first_approval']))
                        
                        return approved_drugs
                        
            except Exception as indication_error:
                logger.warning(f"Approved drug indication search failed: {indication_error}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error in _get_approved_drugs_by_disease: {e}")
            return []
    
    def _get_clinical_candidates(self, phase_filter: Optional[int], max_results: int) -> List[Dict[str, Any]]:
        """Get clinical candidates"""
        if phase_filter:
            candidates = list(self.client.molecule.filter(max_phase=phase_filter))
        else:
            # Get all clinical phase compounds (1-3)
            candidates = []
            for phase in [1, 2, 3]:
                phase_compounds = list(self.client.molecule.filter(max_phase=phase))
                candidates.extend(phase_compounds)
        
        return candidates[:max_results]
    
    def _get_disease_drugs(self, disease_term: str, max_results: int) -> List[Dict[str, Any]]:
        """Get drugs for a specific disease"""
        try:
            # Method 1: Direct search on molecule indication_class (fastest)
            drugs = list(self.client.molecule.filter(
                indication_class__icontains=disease_term,
                max_phase__gte=1  # At least in clinical trials
            ).only(['molecule_chembl_id', 'pref_name', 'max_phase', 'indication_class', 'molecule_properties'])[:max_results])
            
            if drugs:
                return drugs
            
            # Method 2: Use drug_indication approach (like your example) but with limits
            try:
                # Get drug indications with limit - don't fetch all
                indications = list(self.client.drug_indication.filter(
                    efo_term__icontains=disease_term
                ).only(['molecule_chembl_id'])[:100])  # Limit to first 100 indications
                
                if indications:
                    # Extract molecule IDs, filter out None values
                    molecule_ids = [
                        ind['molecule_chembl_id'] for ind in indications 
                        if ind.get('molecule_chembl_id')
                    ]
                    
                    # Remove duplicates and limit
                    unique_molecule_ids = list(set(molecule_ids))[:max_results]
                    
                    if unique_molecule_ids:
                        # Batch fetch molecules using __in filter (like your example)
                        drugs = list(self.client.molecule.filter(
                            molecule_chembl_id__in=unique_molecule_ids
                        ).only(['molecule_chembl_id', 'pref_name', 'max_phase', 'indication_class', 'molecule_properties']))
                        
                        return drugs
                        
            except Exception as indication_error:
                logger.warning(f"Drug indication search failed: {indication_error}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error in _get_disease_drugs: {e}")
            return []
    
    def _get_target_drugs(self, target_query: str, max_results: int) -> List[Dict[str, Any]]:
        """Get drugs for a specific target"""
        try:
            # Find targets
            targets = self.client.search_targets_by_gene(target_query)
            if not targets:
                return []
            
            target_id = targets[0]['target_chembl_id']
            
            # Get activities for this target with better filtering
            activities = list(self.client.activity.filter(
                target_chembl_id=target_id,
                pchembl_value__isnull=False  # Only activities with pChEMBL values
            ).only(['molecule_chembl_id', 'pchembl_value', 'standard_type'])[:max_results * 2])  # Get more activities to filter from
            
            # Get unique molecules from activities, prioritize high-affinity compounds
            molecule_data = {}
            for act in activities:
                mol_id = act.get('molecule_chembl_id')
                pchembl = act.get('pchembl_value')
                if mol_id and pchembl:
                    if mol_id not in molecule_data or pchembl > molecule_data[mol_id]:
                        molecule_data[mol_id] = pchembl
            
            # Sort by pChEMBL value and take top compounds
            top_molecules = sorted(molecule_data.items(), key=lambda x: x[1], reverse=True)[:max_results]
            molecule_ids = [mol_id for mol_id, _ in top_molecules]
            
            # Batch fetch molecules
            if molecule_ids:
                drugs = list(self.client.molecule.filter(
                    molecule_chembl_id__in=molecule_ids,
                    max_phase__gte=1  # At least in clinical development
                ).only(['molecule_chembl_id', 'pref_name', 'max_phase', 'indication_class', 'molecule_properties']))
            else:
                drugs = []
            
            return drugs
            
        except Exception as e:
            logger.error(f"Error in _get_target_drugs: {e}")
            return []
    
    def _get_similar_drugs(self, query_compound: str, similarity_threshold: int, max_results: int) -> List[Dict[str, Any]]:
        """Get drugs similar to a given compound"""
        # Determine if query is SMILES or ChEMBL ID
        if query_compound.startswith('CHEMBL'):
            similar_compounds = self.client.find_similar_compounds(
                chembl_id=query_compound, similarity=similarity_threshold
            )
        else:
            # Assume SMILES
            similar_compounds = self.client.find_similar_compounds(
                smiles=query_compound, similarity=similarity_threshold
            )
        
        drugs = []
        for comp in similar_compounds[:max_results]:
            mol_data = self.client.get_molecule_by_chembl_id(comp['molecule_chembl_id'])
            if mol_data and mol_data.get('max_phase', 0) > 0:  # Filter for compounds with some development
                mol_data['similarity'] = comp.get('similarity')
                drugs.append(mol_data)
        
        return drugs
    
    def _process_drug_results(self, results: List[Dict[str, Any]], include_properties: bool) -> List[Dict[str, Any]]:
        """Process drug results for output"""
        processed_results = []
        
        for result in results:
            # Skip None results
            if not result:
                continue
                
            processed_result = {
                "molecule_chembl_id": result.get("molecule_chembl_id"),
                "pref_name": result.get("pref_name"),
                "max_phase": result.get("max_phase"),
                "first_approval": result.get("first_approval"),
                "indication_class": result.get("indication_class"),
                "therapeutic_flag": result.get("therapeutic_flag"),
                "oral": result.get("oral"),
                "parenteral": result.get("parenteral"),
                "topical": result.get("topical")
            }
            
            # Add similarity if present
            if "similarity" in result:
                processed_result["similarity"] = result["similarity"]
            
            # Add molecular properties if requested
            if include_properties and result.get("molecule_properties"):
                props = result["molecule_properties"]
                processed_result["properties"] = {
                    "molecular_weight": props.get("mw_freebase"),
                    "logp": props.get("alogp"),
                    "hbd": props.get("hbd"),
                    "hba": props.get("hba"),
                    "psa": props.get("psa"),
                    "rotatable_bonds": props.get("rtb"),
                    "ro5_violations": props.get("num_ro5_violations"),
                    "qed_weighted": props.get("qed_weighted")
                }
            
            # Add structure info if available
            if result.get("molecule_structures"):
                structures = result["molecule_structures"]
                processed_result["smiles"] = structures.get("canonical_smiles")
                processed_result["inchi_key"] = structures.get("standard_inchi_key")
            
            # Add development info
            if result.get("withdrawn_flag"):
                processed_result["withdrawn"] = {
                    "withdrawn_flag": result.get("withdrawn_flag"),
                    "withdrawn_reason": result.get("withdrawn_reason"),
                    "withdrawn_year": result.get("withdrawn_year")
                }
            
            # Only add if we have essential data
            if processed_result.get("molecule_chembl_id"):
                processed_results.append(processed_result)
        
        return processed_results
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version of the tool"""
        return self._run(*args, **kwargs)