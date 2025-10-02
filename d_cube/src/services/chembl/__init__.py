"""
ChEMBL webresource client tools for drug discovery and compound analysis
"""

from .utils import ChEMBLClient
from .compound_search import ChEMBLCompoundSearchTool
from .bioactivity_search import ChEMBLBioactivityTool
from .drug_discovery import ChEMBLDrugDiscoveryTool

__all__ = [
    'ChEMBLClient',
    'ChEMBLCompoundSearchTool',
    'ChEMBLBioactivityTool', 
    'ChEMBLDrugDiscoveryTool'
]

# For easy access to all ChEMBL tools
def get_chembl_tools():
    """Get all ChEMBL tools for use in the agent"""
    return [
        ChEMBLCompoundSearchTool(),
        ChEMBLBioactivityTool(),
        ChEMBLDrugDiscoveryTool(),
    ]