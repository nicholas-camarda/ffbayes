#!/usr/bin/env python3
"""
Validate that a draft team exists and is properly formatted.
This script must pass before any post-draft analysis can run.
"""

import os
from datetime import datetime

import pandas as pd


def validate_draft_team():
    """Validate that a draft team exists and is properly formatted."""
    print("🔍 Validating draft team...")
    
    current_year = datetime.now().year
    from ffbayes.utils.path_constants import get_teams_dir
    team_file = str(get_teams_dir() / f"drafted_team_{current_year}.tsv")
    
    # Check if team file exists
    if not os.path.exists(team_file):
        raise FileNotFoundError(
            f"❌ Draft team file not found: {team_file}\n"
            "💡 You must draft players first before running post-draft analysis.\n"
            "📝 Fill in your drafted players in the team file, then run the post-draft pipeline."
        )
    
    # Load and validate team data
    try:
        team_df = pd.read_csv(team_file, sep='\t')
        print(f"📊 Loaded team file: {team_file}")
        print(f"   Players found: {len(team_df)}")
        
        # Check required columns - support both formats
        if 'PLAYER' in team_df.columns and 'POS' in team_df.columns:
            # User's actual format: POS, PLAYER, BYE
            print("✅ Using user's team file format: POS, PLAYER, BYE")
            player_col = 'PLAYER'
            position_col = 'POS'
        elif 'Name' in team_df.columns and 'Position' in team_df.columns:
            # Standard format: Name, Position, Team
            print("✅ Using standard team file format: Name, Position, Team")
            player_col = 'Name'
            position_col = 'Position'
        else:
            raise ValueError(
                f"❌ Unsupported column format\n"
                f"📋 Expected: POS/PLAYER/BYE OR Name/Position/Team\n"
                f"📊 Found columns: {list(team_df.columns)}"
            )
        
        # Check if team has actual players (not just headers or empty rows)
        if len(team_df) == 0:
            raise ValueError(
                "❌ Team file is empty - no players found.\n"
                "💡 Add your drafted players to the team file."
            )
        
        # Check for empty player names
        empty_names = team_df[team_df[player_col].isna() | (team_df[player_col] == '')]
        if len(empty_names) > 0:
            raise ValueError(
                f"❌ Found {len(empty_names)} players with empty names.\n"
                "💡 All players must have valid names."
            )
        
        # Check for empty positions
        empty_positions = team_df[team_df[position_col].isna() | (team_df[position_col] == '')]
        if len(empty_positions) > 0:
            raise ValueError(
                f"❌ Found {len(empty_positions)} players with empty positions.\n"
                "💡 All players must have valid positions."
            )
        
        # Validate position values - support user's format including FLEX, BE, D/ST
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF', 'FLEX', 'BE', 'D/ST']
        invalid_positions = team_df[~team_df[position_col].isin(valid_positions)]
        if len(invalid_positions) > 0:
            print(f"⚠️  Warning: Found {len(invalid_positions)} players with unusual positions:")
            for _, row in invalid_positions.iterrows():
                print(f"   {row[player_col]}: {row[position_col]}")
        
        # Show team summary
        print("✅ Draft team validation passed!")
        print("📊 Team composition:")
        position_counts = team_df[position_col].value_counts()
        for pos, count in position_counts.items():
            print(f"   {pos}: {count}")
        
        # Check if team size is reasonable (typically 15-18 players)
        if len(team_df) < 10:
            print(f"⚠️  Warning: Team only has {len(team_df)} players - this seems small for a fantasy team.")
        elif len(team_df) > 25:
            print(f"⚠️  Warning: Team has {len(team_df)} players - this seems large for a fantasy team.")
        
        # Validate player names for database compatibility
        try:
            from ffbayes.utils.name_resolver import create_name_resolver
            resolver = create_name_resolver()
            
            # Convert to standard format for name resolution
            if player_col == 'PLAYER':
                temp_df = team_df.copy()
                temp_df['Name'] = temp_df[player_col]
            else:
                temp_df = team_df.copy()
            
            print("\n🔍 Validating player names for database compatibility...")
            resolved_df, resolution_log = resolver.resolve_team_names(temp_df)
            
            # Print resolution summary
            resolver.print_resolution_summary(resolution_log)
            
            # Check for potential issues
            unresolved_count = sum(1 for r in resolution_log if r['method'] == 'unresolved')
            if unresolved_count > 0:
                print(f"\n⚠️  Warning: {unresolved_count} player names may not be found in the database")
                print("   This could cause issues in Monte Carlo validation")
                print("   Consider updating these names to match the database format")
            
            # Save resolution log for reference (main results directory, no timestamp)
            from ffbayes.utils.path_constants import get_post_draft_dir
            current_year = datetime.now().year
            post_draft_dir = get_post_draft_dir(current_year)
            log_path = post_draft_dir / "name_validation_log.csv"
            resolver.save_resolution_log(resolution_log, str(log_path))
            
            # Don't save duplicate team file - Monte Carlo can use the original my_ff_teams file directly

            # No need to save a duplicate team file - Monte Carlo can use the original my_ff_teams file directly
            
        except Exception as e:
            print(f"⚠️  Warning: Name validation failed: {e}")
            print("   Player names will be validated during Monte Carlo analysis")
        
        return True
        
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"❌ Team file is empty or corrupted: {team_file}\n"
            "💡 Ensure the file contains valid TSV data with headers."
        )
    except Exception as e:
        raise ValueError(f"❌ Error reading team file: {e}")


def main():
    """Main validation function."""
    print("=" * 60)
    print("Draft Team Validation")
    print("=" * 60)
    
    try:
        validate_draft_team()
        print("\n🎉 Validation successful! Ready for post-draft analysis.")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        print("\n💡 To fix this:")
        print("   1. Complete your fantasy football draft")
        print("   2. Fill in your drafted players in your team file")
        print("   3. Ensure columns are: POS, PLAYER, BYE (your format)")
        print("   4. Run this validation again")
        raise


if __name__ == "__main__":
    main()
