from typing import Optional, Type, Dict, Any
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CSVLoaderSchema(BaseModel):
    """Input schema for CSV loading."""
    
    file_path: str = Field(
        ...,
        description="Path to the CSV file to load"
    )
    max_rows: Optional[int] = Field(
        10,
        description="Maximum number of rows to preview (default: 10)"
    )


class CSVDataLoaderTool(BaseTool):
    """Simple tool for loading CSV datasets."""
    
    name: str = "load_csv_dataset"
    description: str = (
        "Load and preview CSV datasets. Shows basic info about the dataset "
        "including columns, data types, and first few rows."
    )
    args_schema: Type[CSVLoaderSchema] = CSVLoaderSchema

    def _run(
        self,
        file_path: str,
        max_rows: Optional[int] = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute CSV loading."""
        
        try:
            # Check if file exists
            path = Path(file_path)
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}"
                }
            
            # Load CSV
            logger.info(f"Loading CSV: {file_path}")
            df = pd.read_csv(file_path)
            
            # Get basic info
            result = {
                "success": True,
                "file_path": file_path,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "preview": df.head(max_rows).to_dict('records'),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }