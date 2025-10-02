# src/services/chembl/utils.py
"""
Base utilities for ChEMBL webresource client operations
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils

logger = logging.getLogger(__name__)

class ChEMBLClient:
    """Base class for ChEMBL webresource client operations"""
    
    def __init__(self):
        """Initialize ChEMBL client resources"""
        self.molecule = new_client.molecule
        self.activity = new_client.activity
        self.target = new_client.target
        self.assay = new_client.assay
        self.document = new_client.document
        self.drug = new_client.drug
        self.drug_indication = new_client.drug_indication
        self.similarity = new_client.similarity
        self.image = new_client.image
        self.tissue = new_client.tissue
        self.cell_line = new_client.cell_line
        self.source = new_client.source
        
    def search_molecule_by_name(self, name: str, exact: bool = True) -> List[Dict[str, Any]]:
        """Search for molecules by preferred name or synonyms"""
        try:
            if exact:
                # First try exact match on pref_name
                results = list(self.molecule.filter(pref_name__iexact=name))
                if results:
                    return results
                
                # If no exact match, try synonyms
                results = list(self.molecule.filter(
                    molecule_synonyms__molecule_synonym__iexact=name
                ).only('molecule_chembl_id', 'pref_name', 'molecule_structures'))
                return results
            else:
                # Case-insensitive contains search
                results = list(self.molecule.filter(pref_name__icontains=name))
                if not results:
                    results = list(self.molecule.filter(
                        molecule_synonyms__molecule_synonym__icontains=name
                    ).only('molecule_chembl_id', 'pref_name', 'molecule_structures'))
                return results
                
        except Exception as e:
            logger.error(f"Error searching molecule by name '{name}': {e}")
            return []
    
    def get_molecule_by_chembl_id(self, chembl_id: str, include_structures: bool = True) -> Optional[Dict[str, Any]]:
        """Get molecule by ChEMBL ID"""
        try:
            fields = ['molecule_chembl_id', 'pref_name', 'molecule_properties']
            if include_structures:
                fields.append('molecule_structures')
                
            results = list(self.molecule.filter(molecule_chembl_id=chembl_id).only(fields))
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error getting molecule by ChEMBL ID '{chembl_id}': {e}")
            return None
    
    def get_molecules_by_inchi_key(self, inchi_key: str) -> List[Dict[str, Any]]:
        """Get molecules by standard InChI key"""
        try:
            return list(self.molecule.filter(
                molecule_structures__standard_inchi_key=inchi_key
            ).only(['molecule_chembl_id', 'pref_name', 'molecule_structures']))
            
        except Exception as e:
            logger.error(f"Error getting molecule by InChI key '{inchi_key}': {e}")
            return []
    
    def find_similar_compounds(self, smiles: str = None, chembl_id: str = None, 
                             similarity: int = 70) -> List[Dict[str, Any]]:
        """Find compounds similar to given SMILES or ChEMBL ID"""
        try:
            if smiles:
                return list(self.similarity.filter(
                    smiles=smiles, similarity=similarity
                ).only(['molecule_chembl_id', 'similarity']))
            elif chembl_id:
                return list(self.similarity.filter(
                    chembl_id=chembl_id, similarity=similarity
                ).only(['molecule_chembl_id', 'pref_name', 'similarity']))
            else:
                raise ValueError("Either smiles or chembl_id must be provided")
                
        except Exception as e:
            logger.error(f"Error finding similar compounds: {e}")
            return []
    
    def search_targets_by_gene(self, gene_name: str) -> List[Dict[str, Any]]:
        """Search targets by gene name"""
        try:
            return list(self.target.filter(
                target_synonym__icontains=gene_name
            ).only(['target_chembl_id', 'organism', 'pref_name', 'target_type']))
            
        except Exception as e:
            logger.error(f"Error searching targets by gene '{gene_name}': {e}")
            return []
    
    def get_activities_for_target(self, target_chembl_id: str, 
                                activity_type: str = None,
                                assay_type: str = None) -> List[Dict[str, Any]]:
        """Get activities for a specific target"""
        try:
            filters = {'target_chembl_id': target_chembl_id}
            
            if activity_type:
                filters['standard_type'] = activity_type
            if assay_type:
                filters['assay_type'] = assay_type
                
            return list(self.activity.filter(**filters))
            
        except Exception as e:
            logger.error(f"Error getting activities for target '{target_chembl_id}': {e}")
            return []
    
    def get_activities_for_molecule(self, molecule_chembl_id: str, 
                                  with_pchembl: bool = False) -> List[Dict[str, Any]]:
        """Get activities for a specific molecule"""
        try:
            filters = {'molecule_chembl_id': molecule_chembl_id}
            
            if with_pchembl:
                filters['pchembl_value__isnull'] = False
                
            return list(self.activity.filter(**filters))
            
        except Exception as e:
            logger.error(f"Error getting activities for molecule '{molecule_chembl_id}': {e}")
            return []
    
    def filter_molecules_by_properties(self, 
                                     max_mw: float = None,
                                     min_mw: float = None,
                                     max_logp: float = None,
                                     min_logp: float = None,
                                     ro5_violations: int = None,
                                     name_pattern: str = None) -> List[Dict[str, Any]]:
        """Filter molecules by molecular properties"""
        try:
            filters = {}
            
            if max_mw is not None:
                filters['molecule_properties__mw_freebase__lte'] = max_mw
            if min_mw is not None:
                filters['molecule_properties__mw_freebase__gte'] = min_mw
            if max_logp is not None:
                filters['molecule_properties__alogp__lte'] = max_logp
            if min_logp is not None:
                filters['molecule_properties__alogp__gte'] = min_logp
            if ro5_violations is not None:
                filters['molecule_properties__num_ro5_violations'] = ro5_violations
            if name_pattern:
                filters['pref_name__icontains'] = name_pattern
                
            return list(self.molecule.filter(**filters).only([
                'molecule_chembl_id', 'pref_name', 'molecule_properties'
            ]))
            
        except Exception as e:
            logger.error(f"Error filtering molecules by properties: {e}")
            return []
    
    def get_approved_drugs(self, indication_contains: str = None,
                          approval_year_gte: int = None) -> List[Dict[str, Any]]:
        """Get approved drugs (max_phase=4)"""
        try:
            filters = {'max_phase': 4}
            
            if approval_year_gte:
                filters['first_approval__gte'] = approval_year_gte
                
            results = list(self.molecule.filter(**filters))
            
            # Filter by indication if specified
            if indication_contains:
                filtered_results = []
                for drug in results:
                    if (drug.get('indication_class') and 
                        indication_contains.lower() in drug['indication_class'].lower()):
                        filtered_results.append(drug)
                return filtered_results
                
            return results
            
        except Exception as e:
            logger.error(f"Error getting approved drugs: {e}")
            return []
    
    def search_drug_indications(self, indication_term: str) -> List[Dict[str, Any]]:
        """Search drug indications by term"""
        try:
            return list(self.drug_indication.filter(efo_term__icontains=indication_term))
            
        except Exception as e:
            logger.error(f"Error searching drug indications for '{indication_term}': {e}")
            return []
    
    # Utility methods for molecular processing
    def standardize_molecule(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Standardize a molecule using ChEMBL utils"""
        try:
            mol_ctab = utils.smiles2ctab(smiles)
            standardized = json.loads(utils.standardize(mol_ctab))
            return standardized[0] if standardized else None
            
        except Exception as e:
            logger.error(f"Error standardizing molecule '{smiles}': {e}")
            return None
    
    def calculate_descriptors(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Calculate molecular descriptors"""
        try:
            mol_ctab = utils.smiles2ctab(smiles)
            descriptors = json.loads(utils.chemblDescriptors(mol_ctab))
            return descriptors[0] if descriptors else None
            
        except Exception as e:
            logger.error(f"Error calculating descriptors for '{smiles}': {e}")
            return None
    
    def get_structural_alerts(self, smiles: str) -> List[Dict[str, Any]]:
        """Get structural alerts for a molecule"""
        try:
            mol_ctab = utils.smiles2ctab(smiles)
            alerts = json.loads(utils.structuralAlerts(mol_ctab))
            return alerts[0] if alerts else []
            
        except Exception as e:
            logger.error(f"Error getting structural alerts for '{smiles}': {e}")
            return []
    
    def get_parent_molecule(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Get parent molecule (removes salts, etc.)"""
        try:
            mol_ctab = utils.smiles2ctab(smiles)
            parent = json.loads(utils.getParent(mol_ctab))
            return parent[0] if parent else None
            
        except Exception as e:
            logger.error(f"Error getting parent molecule for '{smiles}': {e}")
            return None