# src/services/cheminformatics/conformer_tool.py
"""
Tool for conformational sampling of molecules
"""
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem.rdMolAlign import AlignMol
    import numpy as np
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConformerInput(BaseModel):
    """Input for conformational sampling"""
    smiles_list: List[str] = Field(description="List of SMILES strings to generate conformers for")
    num_conformers: int = Field(default=10, description="Number of conformers to generate per molecule")
    energy_window: float = Field(default=10.0, description="Energy window in kcal/mol for conformer pruning")
    rms_threshold: float = Field(default=0.5, description="RMS threshold for conformer pruning")
    optimization_steps: int = Field(default=200, description="Number of optimization steps")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

class ConformerTool(BaseTool):
    """Tool for generating molecular conformers"""
    
    name: str = "conformational_sampling"
    description: str = """
    Generate multiple conformers for molecules using RDKit's ETKDG algorithm.
    
    Performs:
    - Distance geometry conformer generation (ETKDG)
    - Energy minimization with MMFF94 force field
    - Conformer pruning based on energy and RMS diversity
    - 3D coordinate generation for molecular modeling
    
    Essential for structure-based drug design and molecular dynamics preparation.
    """
    args_schema: type[BaseModel] = ConformerInput
    
    def _run(self, 
             smiles_list: List[str],
             num_conformers: int = 10,
             energy_window: float = 10.0,
             rms_threshold: float = 0.5,
             optimization_steps: int = 200,
             random_seed: int = 42) -> str:
        """Generate conformers for molecules"""
        
        if not RDKIT_AVAILABLE:
            return json.dumps({
                "success": False,
                "error": "RDKit not available. Install with: pip install rdkit"
            })
        
        try:
            results = []
            total_conformers = 0
            
            for i, smiles in enumerate(smiles_list):
                try:
                    # Parse SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results.append({
                            "input_smiles": smiles,
                            "input_index": i,
                            "success": False,
                            "error": "Invalid SMILES",
                            "conformers": []
                        })
                        continue
                    
                    # Add hydrogens
                    mol = Chem.AddHs(mol)
                    
                    # Generate conformers using ETKDG
                    params = AllChem.ETKDGv3()
                    params.randomSeed = random_seed
                    # params.maxAttempts = num_conformers * 10
                    params.pruneRmsThresh = rms_threshold
                    params.useExpTorsionAnglePrefs = True
                    params.useBasicKnowledge = True
                    
                    # Generate initial conformers
                    conf_ids = AllChem.EmbedMultipleConfs(
                        mol, 
                        numConfs=num_conformers,
                        params=params
                    )
                    
                    if not conf_ids:
                        results.append({
                            "input_smiles": smiles,
                            "input_index": i,
                            "success": False,
                            "error": "No conformers generated",
                            "conformers": []
                        })
                        continue
                    
                    # Optimize conformers with MMFF94
                    energies = []
                    for conf_id in conf_ids:
                        try:
                            # MMFF optimization
                            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=optimization_steps)
                            
                            # Calculate energy
                            ff = AllChem.MMFFGetMoleculeForceField(mol, confId=conf_id)
                            if ff:
                                energy = ff.CalcEnergy()
                                energies.append((conf_id, energy))
                        except:
                            continue
                    
                    if not energies:
                        results.append({
                            "input_smiles": smiles,
                            "input_index": i,
                            "success": False,
                            "error": "Energy calculation failed",
                            "conformers": []
                        })
                        continue
                    
                    # Sort by energy and apply energy window
                    energies.sort(key=lambda x: x[1])
                    min_energy = energies[0][1]
                    filtered_conformers = [
                        (conf_id, energy) for conf_id, energy in energies 
                        if energy - min_energy <= energy_window
                    ]
                    
                    # Process conformer data
                    conformers = []
                    for j, (conf_id, energy) in enumerate(filtered_conformers):
                        conf = mol.GetConformer(conf_id)
                        
                        # Get molecular properties
                        mol_props = {
                            "molecular_weight": rdMolDescriptors.CalcExactMolWt(mol),
                            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                            "heavy_atoms": mol.GetNumHeavyAtoms()
                        }
                        
                        conformers.append({
                            "conformer_id": j,
                            "rdkit_conf_id": int(conf_id),
                            "energy_kcal_mol": round(energy, 4),
                            "relative_energy": round(energy - min_energy, 4),
                            "num_atoms": mol.GetNumAtoms(),
                            "molecular_properties": mol_props
                        })
                    
                    results.append({
                        "input_smiles": smiles,
                        "input_index": i,
                        "success": True,
                        "num_conformers": len(conformers),
                        "energy_range": round(max(c["energy_kcal_mol"] for c in conformers) - min_energy, 4),
                        "conformers": conformers,
                        "mol_object_available": True  # For SDF export
                    })
                    
                    total_conformers += len(conformers)
                    
                except Exception as e:
                    logger.error(f"Error processing SMILES {smiles}: {e}")
                    results.append({
                        "input_smiles": smiles,
                        "input_index": i,
                        "success": False,
                        "error": str(e),
                        "conformers": []
                    })
            
            success_count = sum(1 for r in results if r["success"])
            
            return json.dumps({
                "success": True,
                "processed_count": len(smiles_list),
                "success_count": success_count,
                "total_conformers": total_conformers,
                "parameters": {
                    "num_conformers_requested": num_conformers,
                    "energy_window_kcal_mol": energy_window,
                    "rms_threshold": rms_threshold,
                    "optimization_steps": optimization_steps
                },
                "results": results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Conformer generation error: {e}")
            return json.dumps({
                "success": False,
                "error": f"Conformer generation failed: {str(e)}"
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version of the tool"""
        return self._run(*args, **kwargs)