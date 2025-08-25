#!/usr/bin/env python3
"""
Name Resolver - Normalize player names for consistent matching across datasets.
Handles initials, suffixes, punctuation, nicknames, and fuzzy matching.
"""

import csv
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class NameResolver:
    """Resolve and normalize player names for consistent matching."""
    
    def __init__(self, unified_dataset_path: Optional[str] = None):
        """
        Initialize name resolver with unified dataset.
        
        Args:
            unified_dataset_path: Path to unified dataset JSON file
        """
        self.unified_dataset_path = unified_dataset_path
        self.name_lookup = {}
        self.reverse_lookup = {}
        self.fuzzy_threshold = 0.95  # Very high threshold to avoid wrong matches
        self.name_position_lookup = {}  # Store name -> position mapping
        
        # Common name patterns and mappings
        self.suffix_patterns = [
            r'\s+Jr\.?$', r'\s+Sr\.?$', r'\s+I{2,4}$', r'\s+IV$', r'\s+V$',
            r'\s+VI$', r'\s+VII$', r'\s+VIII$', r'\s+IX$', r'\s+X$'
        ]
        
        # Common nickname mappings
        self.nickname_mappings = {
            'P. Mahomes': 'Patrick Mahomes',
            'T. Brady': 'Tom Brady',
            'A. Rodgers': 'Aaron Rodgers',
            'D. Prescott': 'Dak Prescott',
            'J. Allen': 'Josh Allen',
            'L. Jackson': 'Lamar Jackson',
            'J. Herbert': 'Justin Herbert',
            'T. Lawrence': 'Trevor Lawrence',
            'B. Purdy': 'Brock Purdy',
            'K. Murray': 'Kyler Murray',
            'B. Mayfield': 'Baker Mayfield',
            'D. Jones': 'Daniel Jones',
            'M. Stafford': 'Matthew Stafford',
            'J. Goff': 'Jared Goff',
            'R. Wilson': 'Russell Wilson',
            'J. Burrow': 'Joe Burrow',
            'T. Tagovailoa': 'Tua Tagovailoa',
            'C. Stroud': 'C.J. Stroud',
            'B. Young': 'Bryce Young',
            'A. Richardson': 'Anthony Richardson',
            'W. Levis': 'Will Levis',
            'J. McCarthy': 'J.J. McCarthy',
            'S. Rattler': 'Spencer Rattler',
            'M. Nabers': 'Malik Nabers',
            'R. Odunze': 'Rome Odunze',
            'B. Bowers': 'Brock Bowers',
            'C. Williams': 'Caleb Williams',
            'D. Maye': 'Drake Maye',
            'J. Daniels': 'Jayden Daniels',
            'M. Penix Jr.': 'Michael Penix Jr.',
            'B. Nix': 'Bo Nix',
            'A. St. Brown': 'Amon-Ra St. Brown',
            'D. Kincaid': 'Dalton Kincaid',
            'J. Cook': 'James Cook',
            'K. Coleman': 'Keon Coleman',
            'M. Pittman Jr.': 'Michael Pittman Jr.',
            'T. Hockenson': 'T.J. Hockenson',
            'C. Hubbard': 'Chuba Hubbard',
            'J. Jeudy': 'Jerry Jeudy',
            'N. Chubb': 'Nick Chubb',
            'A. Thielen': 'Adam Thielen',
        }
        
        # Load unified dataset if provided
        if unified_dataset_path and os.path.exists(unified_dataset_path):
            self._build_name_lookup()
    
    def _build_name_lookup(self):
        """Build name lookup from unified dataset."""
        try:
            # Load unified dataset
            with open(self.unified_dataset_path, 'r') as f:
                import json
                data = json.load(f)
            
            # Extract unique player names
            unique_names = set()
            for record in data:
                if 'Name' in record and record['Name']:
                    unique_names.add(record['Name'].strip())
            
            # Build lookup dictionaries
            for record in data:
                if 'Name' in record and record['Name']:
                    name = record['Name'].strip()
                    
                    # Store original name
                    self.name_lookup[name] = name
                    
                    # Store normalized versions
                    normalized = self._normalize_name(name)
                    if normalized != name:
                        self.name_lookup[normalized] = name
                    
                    # Store without suffixes
                    no_suffix = self._remove_suffixes(name)
                    if no_suffix != name:
                        self.name_lookup[no_suffix] = name
                    
                    # Store initials version
                    initials = self._name_to_initials(name)
                    if initials != name:
                        self.name_lookup[initials] = name
                    
                    # Store position mapping if available
                    if 'Position' in record:
                        self.name_position_lookup[name] = record['Position']
            
            logger.info(f"Built name lookup with {len(unique_names)} unique names")
            
        except Exception as e:
            logger.warning(f"Failed to build name lookup: {e}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        if not name:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = name.lower().strip()
        
        # Remove punctuation except apostrophes
        normalized = re.sub(r'[^\w\s\']', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _remove_suffixes(self, name: str) -> str:
        """Remove common name suffixes."""
        for pattern in self.suffix_patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        return name.strip()
    
    def _name_to_initials(self, name: str) -> str:
        """Convert full name to initials format."""
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {parts[1]}"
        return name
    
    def _is_initials_format(self, name: str) -> bool:
        """Check if name is already in initials format (e.g., 'K. Johnson')."""
        parts = name.split()
        if len(parts) == 2:
            return len(parts[0]) == 2 and parts[0].endswith('.')
        return False
    
    def _fuzzy_match(self, name: str, candidates: List[str], position: Optional[str] = None) -> Optional[str]:
        """Find best fuzzy match for a name, optionally requiring position match."""
        best_match = None
        best_ratio = 0
        
        normalized_name = self._normalize_name(name)
        
        for candidate in candidates:
            normalized_candidate = self._normalize_name(candidate)
            ratio = SequenceMatcher(None, normalized_name, normalized_candidate).ratio()
            
            # If position is provided, require position match for fuzzy matching
            if position and candidate in self.name_position_lookup:
                candidate_position = self.name_position_lookup[candidate]
                if position != candidate_position:
                    continue  # Skip if positions don't match
            
            if ratio > best_ratio and ratio >= self.fuzzy_threshold:
                best_ratio = ratio
                best_match = candidate
        
        return best_match
    
    def resolve_name(self, name: str, position: Optional[str] = None) -> Tuple[str, str, float]:
        """
        Resolve a name to its canonical form.
        
        Args:
            name: Name to resolve
            position: Optional position to validate fuzzy matches
            
        Returns:
            Tuple of (resolved_name, resolution_method, confidence)
        """
        if not name or name.strip() == "":
            return name, "empty", 0.0
        
        original_name = name.strip()
        
        # Check nickname mappings FIRST (highest priority)
        if original_name in self.nickname_mappings:
            resolved = self.nickname_mappings[original_name]
            return resolved, "nickname", 0.95
        
        # Check exact match
        if original_name in self.name_lookup:
            return self.name_lookup[original_name], "exact", 1.0
        
        # Check without suffixes
        no_suffix = self._remove_suffixes(original_name)
        if no_suffix in self.name_lookup:
            return self.name_lookup[no_suffix], "no_suffix", 0.9
        
        # Check initials format (only if the original name is already in initials format)
        if self._is_initials_format(original_name):
            if original_name in self.name_lookup:
                return self.name_lookup[original_name], "initials", 0.85
        
        # Check normalized version
        normalized = self._normalize_name(original_name)
        if normalized in self.name_lookup:
            return self.name_lookup[normalized], "normalized", 0.8
        
        # Try fuzzy matching (with position validation if available)
        if self.name_lookup:
            candidates = list(self.name_lookup.keys())
            fuzzy_match = self._fuzzy_match(original_name, candidates, position)
            if fuzzy_match:
                return self.name_lookup[fuzzy_match], "fuzzy", 0.7
        
        # No match found
        return original_name, "unresolved", 0.0
    
    def resolve_team_names(self, team_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Resolve all names in a team DataFrame.
        
        Args:
            team_df: DataFrame with 'Name' column and optional 'Position' column
            
        Returns:
            Tuple of (resolved_dataframe, resolution_log)
        """
        if 'Name' not in team_df.columns:
            return team_df, []
        
        resolved_df = team_df.copy()
        resolution_log = []
        
        for idx, row in team_df.iterrows():
            original_name = row['Name']
            position = row.get('Position') if 'Position' in team_df.columns else None
            resolved_name, method, confidence = self.resolve_name(original_name, position)
            
            resolved_df.at[idx, 'Name'] = resolved_name
            
            resolution_log.append({
                'original_name': original_name,
                'resolved_name': resolved_name,
                'method': method,
                'confidence': confidence,
                'changed': original_name != resolved_name,
                'position': position
            })
        
        return resolved_df, resolution_log
    
    def save_resolution_log(self, resolution_log: List[Dict], output_path: str):
        """Save resolution log to CSV file."""
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'original_name', 'resolved_name', 'method', 'confidence', 'changed', 'position'
                ])
                writer.writeheader()
                writer.writerows(resolution_log)
            
            logger.info(f"Resolution log saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save resolution log: {e}")
    
    def print_resolution_summary(self, resolution_log: List[Dict]):
        """Print a summary of name resolutions."""
        total = len(resolution_log)
        resolved = sum(1 for r in resolution_log if r['changed'])
        unresolved = sum(1 for r in resolution_log if r['method'] == 'unresolved')
        
        print("\n📊 Name Resolution Summary:")
        print(f"   Total names: {total}")
        print(f"   Resolved: {resolved}")
        print(f"   Unresolved: {unresolved}")
        
        if resolved > 0:
            print("\n✅ Successfully resolved names:")
            for log in resolution_log:
                if log['changed']:
                    print(f"   {log['original_name']} → {log['resolved_name']} ({log['method']})")
        
        if unresolved > 0:
            print("\n⚠️  Unresolved names:")
            for log in resolution_log:
                if log['method'] == 'unresolved':
                    print(f"   {log['original_name']}")
        
        # Method breakdown
        method_counts = {}
        for log in resolution_log:
            method = log['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print("\n📈 Resolution methods:")
        for method, count in method_counts.items():
            print(f"   {method}: {count}")


def create_name_resolver() -> NameResolver:
    """Create a name resolver instance with unified dataset."""
    try:
        from ffbayes.utils.path_constants import get_unified_dataset_path
        unified_path = str(get_unified_dataset_path())
        return NameResolver(unified_path)
    except Exception as e:
        logger.warning(f"Failed to create name resolver with unified dataset: {e}")
        return NameResolver()


if __name__ == "__main__":
    # Test the name resolver
    resolver = create_name_resolver()
    
    test_names = [
        "P. Mahomes",
        "M. Pittman Jr.",
        "A. St. Brown",
        "T. Hockenson",
        "Patrick Mahomes",  # Should match exactly
        "Unknown Player",   # Should be unresolved
    ]
    
    print("🧪 Testing Name Resolver")
    print("=" * 50)
    
    for name in test_names:
        resolved, method, confidence = resolver.resolve_name(name)
        print(f"{name:20} → {resolved:20} ({method}, {confidence:.2f})")
