# src/services/cheminformatics/protonation_tool.py
"""
Tool for calculating protonation states of molecules
"""
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from dimorphite_dl import protonate_smiles
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProtonationInput(BaseModel):
    """Input for protonation state calculation"""
    smiles_list: List[str] = Field(description="List of SMILES strings to protonate")
    ph_min: float = Field(default=6.4, description="Minimum pH for protonation")
    ph_max: float = Field(default=8.4, description="Maximum pH for protonation")
    ph_target: Optional[float] = Field(default=7.4, description="Target pH for protonation")
    max_variants: int = Field(default=16, description="Maximum number of protonation variants per molecule")

class ProtonationTool(BaseTool):
    """Tool for calculating protonation states of molecules"""
    
    name: str = "protonation_state_calculator"
    description: str = """
    Calculate protonation states for molecules at physiological pH.
    
    Uses Dimorphite-DL algorithm to:
    - Generate reasonable protonation states at target pH (default 7.4)
    - Handle ionizable groups (acids, bases, ampholytes)
    - Return multiple protonation variants for each input molecule
    
    Essential for drug-like molecules that change ionization state at biological pH.
    """
    args_schema: type[BaseModel] = ProtonationInput
    
    def _run(self, 
             smiles_list: List[str],
             ph_min: float = 6.4,
             ph_max: float = 8.4,
             ph_target: Optional[float] = 7.4,
             max_variants: int = 16) -> str:
        """Calculate protonation states"""
        
        if not RDKIT_AVAILABLE:
            return json.dumps({
                "success": False,
                "error": "RDKit and dimorphite-dl not available. Install with: pip install rdkit dimorphite-dl"
            })
        
        try:
            results = []
            total_variants = 0
            
            for i, smiles in enumerate(smiles_list):
                try:
                    # Validate SMILES first
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results.append({
                            "input_smiles": smiles,
                            "input_index": i,
                            "success": False,
                            "error": "Invalid SMILES",
                            "protonation_states": []
                        })
                        continue
                    
                    # Calculate protonation states using the correct API
                    protonated_smiles = protonate_smiles(
                        smiles,
                        ph_min=ph_min,
                        ph_max=ph_max,
                        precision=0.5,
                        max_variants=max_variants,
                        label_states=True
                    )
                    
                    # Process results
                    protonation_states = []
                    for j, prot_smiles in enumerate(protonated_smiles):
                        try:
                            prot_mol = Chem.MolFromSmiles(prot_smiles)
                            if prot_mol:
                                # Calculate properties
                                charge = Chem.rdmolops.GetFormalCharge(prot_mol)
                                mw = rdMolDescriptors.CalcExactMolWt(prot_mol)
                                
                                protonation_states.append({
                                    "variant_id": j,
                                    "smiles": prot_smiles,
                                    "formal_charge": charge,
                                    "molecular_weight": round(mw, 4),
                                    "ph_range": f"{ph_min}-{ph_max}"
                                })
                        except Exception as e:
                            logger.warning(f"Error processing protonated SMILES {prot_smiles}: {e}")
                            continue
                    
                    results.append({
                        "input_smiles": smiles,
                        "input_index": i,
                        "success": True,
                        "num_variants": len(protonation_states),
                        "protonation_states": protonation_states
                    })
                    
                    total_variants += len(protonation_states)
                    
                except Exception as e:
                    logger.error(f"Error processing SMILES {smiles}: {e}")
                    results.append({
                        "input_smiles": smiles,
                        "input_index": i,
                        "success": False,
                        "error": str(e),
                        "protonation_states": []
                    })
            
            success_count = sum(1 for r in results if r["success"])
            
            return json.dumps({
                "success": True,
                "processed_count": len(smiles_list),
                "success_count": success_count,
                "total_variants": total_variants,
                "ph_conditions": {
                    "ph_min": ph_min,
                    "ph_max": ph_max,
                    "ph_target": ph_target,
                    "precision": 0.5
                },
                "results": results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Protonation calculation error: {e}")
            return json.dumps({
                "success": False,
                "error": f"Protonation calculation failed: {str(e)}"
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version of the tool"""
        return self._run(*args, **kwargs)