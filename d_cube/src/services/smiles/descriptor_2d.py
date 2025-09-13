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

from src.services.smiles.utils import BaseSmilesService

class Descriptors2DSchema(BaseModel):
    """Input schema for 2D descriptor calculation."""
    
    smiles: Union[str, List[str]] = Field(
        ...,
        description="SMILES string(s) to calculate descriptors for."
    )
    descriptor_set: str = Field(
        "lipinski",
        description="Descriptor set to calculate: 'lipinski', 'basic', 'extended', or 'all'"
    )
    include_fingerprints: bool = Field(
        False,
        description="Whether to include Morgan fingerprints in output."
    )
    fingerprint_radius: int = Field(
        2,
        description="Radius for Morgan fingerprints (default: 2)."
    )
    fingerprint_bits: int = Field(
        2048,
        description="Number of bits for Morgan fingerprints (default: 2048)."
    )

    @validator('descriptor_set')
    def validate_descriptor_set(cls, v):
        valid_sets = {'lipinski', 'basic', 'extended', 'all'}
        if v not in valid_sets:
            raise ValueError(f"descriptor_set must be one of: {valid_sets}")
        return v

    @validator('smiles')
    def validate_smiles_input(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("SMILES string cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("SMILES list cannot be empty")
        return v



class Descriptors2DTool(BaseTool):
    """Tool for calculating 2D molecular descriptors."""
    
    name: str = "calculate_2d_descriptors"
    description: str = (
        "Calculate 2D molecular descriptors from SMILES strings. "
        "Supports various descriptor sets including Lipinski parameters, "
        "basic physicochemical properties, and extended descriptor sets."
    )
    args_schema: Type[Descriptors2DSchema] = Descriptors2DSchema

    def _get_descriptor_functions(self, descriptor_set: str) -> Dict[str, callable]:
        """Get descriptor calculation functions based on selected set."""
        
        lipinski_descriptors = {
            "molecular_weight": Descriptors.MolWt,
            "logp": Descriptors.MolLogP,
            "hbd": Descriptors.NumHDonors,  # Hydrogen bond donors
            "hba": Descriptors.NumHAcceptors,  # Hydrogen bond acceptors
            "rotatable_bonds": Descriptors.NumRotatableBonds,
            "tpsa": Descriptors.TPSA,  # Topological polar surface area
        }
        
        basic_descriptors = {
            **lipinski_descriptors,
            "heavy_atom_count": Descriptors.HeavyAtomCount,
            "ring_count": Descriptors.RingCount,
            "aromatic_rings": lambda mol: Descriptors.NumAromaticRings(mol),
            "aliphatic_rings": lambda mol: Descriptors.NumAliphaticRings(mol),
            "formal_charge": Chem.rdmolops.GetFormalCharge,
        }
        
        extended_descriptors = {
            **basic_descriptors,
            "bertz_ct": Descriptors.BertzCT,  # Complexity
            "balaban_j": Descriptors.BalabanJ,  # Topological index
            "chi0v": Descriptors.Chi0v,  # Connectivity indices
            "chi1v": Descriptors.Chi1v,
            "kappa1": Descriptors.Kappa1,  # Kappa shape indices
            "kappa2": Descriptors.Kappa2,
            "kappa3": Descriptors.Kappa3,
        }
        
        descriptor_sets = {
            "lipinski": lipinski_descriptors,
            "basic": basic_descriptors,
            "extended": extended_descriptors,
            "all": extended_descriptors  # Can be extended further
        }
        
        return descriptor_sets.get(descriptor_set, lipinski_descriptors)

    def _calculate_descriptors_single(
        self,
        smiles: str,
        descriptor_set: str,
        include_fingerprints: bool,
        fingerprint_radius: int,
        fingerprint_bits: int
    ) -> Dict[str, Any]:
        """Calculate descriptors for a single SMILES."""
        
        result = {
            "input_smiles": smiles,
            "descriptors": {},
            "fingerprint": None,
            "is_valid": False,
            "error": None
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result["error"] = "Invalid SMILES: Could not parse molecule"
                return result
            
            # Calculate descriptors
            descriptor_functions = self._get_descriptor_functions(descriptor_set)
            descriptors = {}
            
            for desc_name, desc_func in descriptor_functions.items():
                try:
                    value = desc_func(mol)
                    # Round float values for cleaner output
                    if isinstance(value, float):
                        value = round(value, 4)
                    descriptors[desc_name] = value
                except Exception as e:
                    logger.warning(f"Failed to calculate {desc_name} for {smiles}: {e}")
                    descriptors[desc_name] = None
            
            result["descriptors"] = descriptors
            
            # Calculate Morgan fingerprints if requested
            if include_fingerprints:
                try:
                    fp = GetMorganFingerprintAsBitVect(
                        mol, radius=fingerprint_radius, nBits=fingerprint_bits
                    )
                    result["fingerprint"] = list(fp.ToBitString())
                except Exception as e:
                    logger.warning(f"Failed to calculate fingerprint for {smiles}: {e}")
                    result["fingerprint"] = None
            
            result["is_valid"] = True
            
        except Exception as e:
            result["error"] = f"Descriptor calculation failed: {str(e)}"
            logger.error(f"Error calculating descriptors for {smiles}: {e}")
        
        return result

    def _run(
        self,
        smiles: Union[str, List[str]],
        descriptor_set: str = "lipinski",
        include_fingerprints: bool = False,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute 2D descriptor calculation."""
        
        try:
            # Handle single SMILES
            if isinstance(smiles, str):
                result = self._calculate_descriptors_single(
                    smiles, descriptor_set, include_fingerprints,
                    fingerprint_radius, fingerprint_bits
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
                result = self._calculate_descriptors_single(
                    smi, descriptor_set, include_fingerprints,
                    fingerprint_radius, fingerprint_bits
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
            logger.error(f"2D descriptor calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": None
            }