# src/services/cheminformatics/sdf_export_tool.py
"""
Tool for exporting molecules with conformers to SDF files
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

class SDFExportInput(BaseModel):
    """Input for SDF export"""
    molecules_data: List[Dict] = Field(description="List of molecule data with conformers")
    output_directory: str = Field(default="output_sdf", description="Directory to save SDF files")
    individual_files: bool = Field(default=True, description="Create individual SDF files per molecule")
    include_properties: bool = Field(default=True, description="Include molecular properties in SDF")
    filename_prefix: str = Field(default="ligand", description="Prefix for SDF filenames")

class SDFExportTool(BaseTool):
    """Tool for exporting molecules to SDF files"""
    
    name: str = "sdf_export"
    description: str = """
    Export molecules with multiple conformers to SDF files.
    
    Features:
    - Creates individual SDF files per ligand
    - Includes all conformers for each molecule
    - Embeds molecular properties and energies
    - Organized file naming and directory structure
    
    Essential for molecular modeling workflows and structure-based drug design.
    """
    args_schema: type[BaseModel] = SDFExportInput
    
    def _run(self, 
             molecules_data: List[Dict],
             output_directory: str = "output_sdf",
             individual_files: bool = True,
             include_properties: bool = True,
             filename_prefix: str = "ligand") -> str:
        """Export molecules to SDF files"""
        
        if not RDKIT_AVAILABLE:
            return json.dumps({
                "success": False,
                "error": "RDKit not available. Install with: pip install rdkit"
            })
        
        try:
            # Create output directory
            output_path = Path(output_directory)
            output_path.mkdir(exist_ok=True)
            
            export_results = []
            total_files = 0
            total_conformers = 0
            
            for mol_data in molecules_data:
                try:
                    smiles = mol_data.get("input_smiles", "")
                    if not smiles or not mol_data.get("success", False):
                        continue
                    
                    # Recreate molecule from SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    
                    mol = Chem.AddHs(mol)
                    
                    # Re-generate conformers if needed (simplified version)
                    conformers_data = mol_data.get("conformers", [])
                    if not conformers_data:
                        continue
                    
                    # Generate conformers for export
                    AllChem.EmbedMultipleConfs(mol, numConfs=len(conformers_data))
                    
                    # Set up filename
                    mol_index = mol_data.get("input_index", 0)
                    if individual_files:
                        filename = f"{filename_prefix}_{mol_index:03d}.sdf"
                        filepath = output_path / filename
                    else:
                        filename = f"{filename_prefix}_all.sdf"
                        filepath = output_path / filename
                    
                    # Write SDF file
                    writer = Chem.SDWriter(str(filepath))
                    
                    conformer_count = 0
                    for conf_idx, conf_data in enumerate(conformers_data):
                        try:
                            # Get conformer
                            if conf_idx < mol.GetNumConformers():
                                conf_mol = Chem.Mol(mol, confId=conf_idx)
                                
                                # Add properties if requested
                                if include_properties:
                                    conf_mol.SetProp("SMILES", smiles)
                                    conf_mol.SetProp("ConformerID", str(conf_data.get("conformer_id", conf_idx)))
                                    conf_mol.SetProp("Energy_kcal_mol", str(conf_data.get("energy_kcal_mol", "")))
                                    conf_mol.SetProp("RelativeEnergy_kcal_mol", str(conf_data.get("relative_energy", "")))
                                    
                                    # Add molecular properties
                                    mol_props = conf_data.get("molecular_properties", {})
                                    for prop_name, prop_value in mol_props.items():
                                        conf_mol.SetProp(prop_name, str(prop_value))
                                
                                writer.write(conf_mol)
                                conformer_count += 1
                        except Exception as e:
                            logger.warning(f"Error writing conformer {conf_idx}: {e}")
                            continue
                    
                    writer.close()
                    
                    if conformer_count > 0:
                        export_results.append({
                            "molecule_index": mol_index,
                            "smiles": smiles,
                            "filename": filename,
                            "conformers_exported": conformer_count,
                            "file_path": str(filepath),
                            "file_size_bytes": filepath.stat().st_size if filepath.exists() else 0
                        })
                        total_files += 1
                        total_conformers += conformer_count
                    
                except Exception as e:
                    logger.error(f"Error exporting molecule {mol_data.get('input_index', 'unknown')}: {e}")
                    continue
            
            return json.dumps({
                "success": True,
                "output_directory": str(output_path.absolute()),
                "total_files_created": total_files,
                "total_conformers_exported": total_conformers,
                "individual_files": individual_files,
                "export_results": export_results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"SDF export error: {e}")
            return json.dumps({
                "success": False,
                "error": f"SDF export failed: {str(e)}"
            })
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version of the tool"""
        return self._run(*args, **kwargs)