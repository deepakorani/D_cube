from typing import Any
from langchain_core.runnables import run_in_executor
from langchain_core.tools import BaseTool
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import logging

logger = logging.getLogger(__name__)


class BaseSmilesService(BaseTool):
    """Base class for SMILES-related services."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize RDKit standardization tools
        self.uncharger = rdMolStandardize.Uncharger()
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()
    
    async def _arun(self, run_manager, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        # Since SMILES processing doesn't require database tokens like Gmail,
        # we can run directly. Add any needed async preprocessing here.
        return await run_in_executor(None, self._run, *args, **kwargs)
    
    def _remove_salts(self, mol: Chem.Mol) -> Chem.Mol:
        """Remove salt fragments and return the largest fragment."""
        try:
            return self.fragment_chooser.choose(mol)
        except Exception as e:
            logger.warning(f"Salt removal failed: {e}")
            return mol
    
    def _neutralize_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Neutralize charges in the molecule where possible."""
        try:
            return self.uncharger.uncharge(mol)
        except Exception as e:
            logger.warning(f"Neutralization failed: {e}")
            return mol
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate if a SMILES string can be parsed by RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False