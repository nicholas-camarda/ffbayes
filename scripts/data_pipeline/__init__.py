#!/usr/bin/env python3
"""
Data Pipeline Package
Provides access to data collection, validation, and preprocessing functions.
"""

# Import functions from the data collection script
try:
    # Import from the numbered file by adding it to sys.path
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    collect_data_path = os.path.join(current_dir, '01_collect_data.py')
    
    if os.path.exists(collect_data_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("collect_data_01", collect_data_path)
        collect_data_01 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collect_data_01)
        
        # Extract the functions
        check_data_availability = collect_data_01.check_data_availability
        create_dataset = collect_data_01.create_dataset
        process_dataset = collect_data_01.process_dataset
        collect_data_by_year = collect_data_01.collect_data_by_year
        combine_datasets = collect_data_01.combine_datasets
        collect_nfl_data = collect_data_01.collect_nfl_data
    else:
        raise ImportError("01_collect_data.py not found")
except ImportError:
    # If the import fails, define placeholder functions
    def check_data_availability(year):
        """Placeholder function when import fails."""
        raise ImportError("collect_data_01 module not available")
    
    def create_dataset(year):
        """Placeholder function when import fails."""
        raise ImportError("collect_data_01 module not available")
    
    def process_dataset(final_df, year):
        """Placeholder function when import fails."""
        raise ImportError("collect_data_01 module not available")
    
    def collect_data_by_year(year):
        """Placeholder function when import fails."""
        raise ImportError("collect_data_01 module not available")
    
    def combine_datasets(directory_path, output_directory_path, years_to_process):
        """Placeholder function when import fails."""
        raise ImportError("collect_data_01 module not available")
    
    def collect_nfl_data(years=None):
        """Placeholder function when import fails."""
        raise ImportError("collect_data_01 module not available")

# Export the functions
__all__ = [
    'check_data_availability',
    'create_dataset',
    'process_dataset',
    'collect_data_by_year',
    'combine_datasets',
    'collect_nfl_data'
]
