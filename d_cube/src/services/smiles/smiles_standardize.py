from typing import List, Optional, Type, Dict, Any, Union
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, validator
from langchain_core.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.MolStandardize import rdMolStandardize
import logging

logger = logging.getLogger(__name__)


class SmilesStandardizeSchema(BaseModel):
    """Input schema for SMILES standardization."""
    
    smiles: Union[str, List[str]] = Field(
        ...,
        description="SMILES string(s) to standardize. Can be a single SMILES or list of SMILES."
    )
    remove_salts: bool = Field(
        True,
        description="Whether to remove salt fragments and keep largest fragment."
    )
    neutralize: bool = Field(
        True,
        description="Whether to neutralize charges where possible."
    )
    canonical: bool = Field(
        True,
        description="Whether to return canonical SMILES representation."
    )

    @validator('smiles')
    def validate_smiles_input(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("SMILES string cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("SMILES list cannot be empty")
            for smiles in v:
                if not isinstance(smiles, str) or not smiles.strip():
                    raise ValueError("All SMILES in list must be non-empty strings")
        return v

    

class SmilesStandardizeTool(BaseTool):
    """Tool for standardizing SMILES strings."""
    
    name: str = "standardize_smiles"
    description: str = (
        "Standardize SMILES strings by removing salts, neutralizing charges, "
        "and converting to canonical representation. Handles both single SMILES "
        "and batch processing."
    )
    args_schema: Type[SmilesStandardizeSchema] = SmilesStandardizeSchema

    def _standardize_single_smiles(
        self, 
        smiles: str, 
        remove_salts: bool = True,
        neutralize: bool = True,
        canonical: bool = True
    ) -> Dict[str, Any]:
        """Standardize a single SMILES string."""
        result = {
            "input_smiles": smiles,
            "standardized_smiles": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "is_valid": False,
            "error": None
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result["error"] = "Invalid SMILES: Could not parse molecule"
                return result
            
            # Remove salts (keep largest fragment)
            if remove_salts:
                fragment_chooser = rdMolStandardize.LargestFragmentChooser()
                mol = fragment_chooser.choose(mol)
            
            # Neutralize charges
            if neutralize:
                uncharger = rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)
            
            # Generate standardized SMILES
            if canonical:
                standardized = Chem.MolToSmiles(mol, canonical=True)
            else:
                standardized = Chem.MolToSmiles(mol)
            
            result.update({
                "standardized_smiles": standardized,
                "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "is_valid": True
            })
            
        except Exception as e:
            result["error"] = f"Standardization failed: {str(e)}"
            logger.error(f"Error standardizing SMILES {smiles}: {e}")
        
        return result

    def _run(
        self,
        smiles: Union[str, List[str]],
        remove_salts: bool = True,
        neutralize: bool = True,
        canonical: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute SMILES standardization."""
        
        try:
            # Handle single SMILES
            if isinstance(smiles, str):
                result = self._standardize_single_smiles(
                    smiles, remove_salts, neutralize, canonical
                )
                return {
                    "success": True,
                    "results": result,
                    "processed_count": 1,
                    "valid_count": 1 if result["is_valid"] else 0
                }
            
            # Handle batch processing
            results = []
            valid_count = 0
            
            for smi in smiles:
                result = self._standardize_single_smiles(
                    smi, remove_salts, neutralize, canonical
                )
                results.append(result)
                if result["is_valid"]:
                    valid_count += 1
            
            return {
                "success": True,
                "results": results,
                "processed_count": len(smiles),
                "valid_count": valid_count,
                "success_rate": round(valid_count / len(smiles) * 100, 2) if smiles else 0
            }
            
        except Exception as e:
            logger.error(f"SMILES standardization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": None
            }