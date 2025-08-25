#!/usr/bin/env python3
"""
create_human_readable_strategy.py - Convert complex JSON strategy to human-readable formats
Creates Excel spreadsheets and simple text files that are actually usable during a draft.
"""

import json
import os
from datetime import datetime

import pandas as pd


def load_strategy_json(json_path):
    """Load the strategy JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading strategy: {e}")
        return None

def create_draft_cheatsheet(strategy_data, output_dir):
    """Create a simple, draft-ready cheatsheet."""
    cheatsheet = []
    
    for pick_num, pick_data in strategy_data['strategy'].items():
        # Extract pick number (e.g., "Pick 10" -> 10)
        pick_number = int(pick_num.split()[-1])
        
        # Get primary targets (top 3)
        primary = pick_data['primary_targets'][:3]
        backup = pick_data['backup_options'][:3]
        
        # Get position priority
        position_priority = pick_data['position_priority']
        
        # Get reasoning (simplified)
        reasoning = pick_data['reasoning'].split(' - ')[0]  # Just first part
        
        cheatsheet.append({
            'Pick': pick_number,
            'Primary Targets': ' | '.join(primary),
            'Backup Options': ' | '.join(backup),
            'Position Priority': position_priority,
            'Strategy': reasoning,
            'Risk Level': pick_data['uncertainty_analysis']['risk_tolerance']
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(cheatsheet)
    df = df.sort_values('Pick')
    
    # Save as Excel
    excel_path = os.path.join(output_dir, f'DRAFT_CHEATSHEET_POS10_{datetime.now().year}.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Draft Strategy', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Draft Strategy']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"✅ Draft cheatsheet saved: {excel_path}")
    return excel_path

def create_player_rankings(strategy_data, output_dir):
    """Create a simple player rankings list by position."""
    all_players = {}
    
    # Collect all players mentioned in strategy
    for pick_data in strategy_data['strategy'].values():
        for player in pick_data['primary_targets'] + pick_data['backup_options']:
            if player not in all_players:
                all_players[player] = {
                    'Name': player,
                    'Position': None,
                    'Projected Points': None,
                    'Uncertainty': None,
                    'Tier': None
                }
    
    # Try to get position info from metadata if available
    if 'position_scarcity' in strategy_data['metadata']:
        for pos, pos_data in strategy_data['metadata']['position_scarcity'].items():
            # This is a simplified approach - in reality we'd need the actual player data
            pass
    
    # Create simple rankings
    df = pd.DataFrame(list(all_players.values()))
    df = df.sort_values('Name')
    
    # Save as Excel
    excel_path = os.path.join(output_dir, f'PLAYER_RANKINGS_POS10_{datetime.now().year}.xlsx')
    df.to_excel(excel_path, index=False)
    
    print(f"✅ Player rankings saved: {excel_path}")
    return excel_path

def create_simple_text_summary(strategy_data, output_dir):
    """Create a simple text summary for quick reference."""
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("FANTASY FOOTBALL DRAFT STRATEGY - POSITION 10")
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    # Overall strategy
    summary_lines.append("OVERALL STRATEGY:")
    summary_lines.append(f"• Risk Tolerance: {strategy_data['metadata']['risk_tolerance']}")
    summary_lines.append(f"• League Size: {strategy_data['metadata']['league_size']}")
    summary_lines.append(f"• Scoring: {strategy_data['metadata']['scoring_type']}")
    summary_lines.append("")
    
    # Pick-by-pick summary
    summary_lines.append("PICK-BY-PICK STRATEGY:")
    summary_lines.append("-" * 40)
    
    for pick_num, pick_data in strategy_data['strategy'].items():
        pick_number = pick_num.split()[-1]
        primary = pick_data['primary_targets'][:2]  # Top 2 only
        position_priority = pick_data['position_priority']
        
        summary_lines.append(f"Pick {pick_number}: {' | '.join(primary)}")
        summary_lines.append(f"         Priority: {position_priority}")
        summary_lines.append("")
    
    # Save text file
    text_path = os.path.join(output_dir, f'DRAFT_STRATEGY_SUMMARY_POS10_{datetime.now().year}.txt')
    with open(text_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✅ Text summary saved: {text_path}")
    return text_path

def main():
    """Main function to create human-readable outputs."""
    current_year = datetime.now().year
    
    # Input: JSON strategy file (from pre_draft/draft_strategy)
    from ffbayes.utils.strategy_path_generator import \
        get_bayesian_strategy_path
    json_path = get_bayesian_strategy_path()
    
    # Output: Human-readable files in pre_draft folder
    from ffbayes.utils.path_constants import get_pre_draft_dir
    output_dir = str(get_pre_draft_dir(current_year))
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 Converting strategy to human-readable formats...")
    
    # Load strategy
    strategy_data = load_strategy_json(json_path)
    if not strategy_data:
        print("❌ Failed to load strategy data")
        return
    
    # Create human-readable outputs
    try:
        cheatsheet_path = create_draft_cheatsheet(strategy_data, output_dir)
        rankings_path = create_player_rankings(strategy_data, output_dir)
        summary_path = create_simple_text_summary(strategy_data, output_dir)
        
        print("\n✅ Successfully created human-readable outputs:")
        print(f"   📊 Draft Cheatsheet: {cheatsheet_path}")
        print(f"   📋 Player Rankings: {rankings_path}")
        print(f"   📝 Text Summary: {summary_path}")
        print("\n💡 These files are designed to be used during your draft!")
        
    except Exception as e:
        print(f"❌ Error creating outputs: {e}")

if __name__ == "__main__":
    main()
