# Monte Carlo Simulation Implementation Analysis

## Overview
This document analyzes the existing Monte Carlo simulation implementation for fantasy football team projections based on historical data.

## Implementation Details

### File: `scripts/analysis/montecarlo-historical-ff.py`
**Status**: Functional but has recursion issues
**Purpose**: Monte Carlo simulation for team score projections
**Reference**: Based on Scott Rome's approach (https://srome.github.io/Making-Fantasy-Football-Projections-Via-A-Monte-Carlo-Simulation/)

## Core Components

### 1. Data Loading and Preparation

#### `get_combined_data(directory_path)`
**Purpose**: Load and combine all CSV files from datasets directory
**Features**:
- Uses `glob` to find all CSV files
- Concatenates multiple dataframes
- Returns combined pandas dataframe

**Analysis**:
- ✅ Efficient data loading
- ✅ Handles multiple data files
- ⚠️ No error handling for missing files
- ⚠️ No data validation

#### `make_team(team, db)`
**Purpose**: Create team dataframe from draft picks and historical data
**Features**:
- Filters players by team names
- Validates positions (QB, WR, TE, RB)
- Removes duplicates
- Returns filtered dataframe

**Analysis**:
- ✅ Proper team filtering
- ✅ Position validation
- ✅ Duplicate removal
- ✅ Clean data structure

#### `validate_team(db_team, my_team)`
**Purpose**: Check which team members have historical data
**Features**:
- Compares team members with available data
- Reports missing players
- Provides validation feedback

**Analysis**:
- ✅ Team validation
- ✅ Missing player identification
- ✅ Clear reporting
- ✅ Useful for debugging

### 2. Game and Player Scoring

#### `get_games(db, year, week)`
**Purpose**: Get all players for specific year and week
**Features**:
- Filters by season and game number
- Returns subset of database

**Analysis**:
- ✅ Simple and efficient filtering
- ✅ Clear function purpose
- ✅ Returns expected data structure

#### `score_player(p, db, year, week)`
**Purpose**: Get fantasy points for specific player in specific game
**Features**:
- Looks up player by name, season, and game
- Returns fantasy points value

**Analysis**:
- ✅ Direct player scoring
- ✅ Clear lookup logic
- ⚠️ No error handling for missing players
- ⚠️ Assumes player exists in data

### 3. Monte Carlo Simulation Core

#### `get_score_for_player(db, player, years)`
**Purpose**: Sample historical performance for player
**Features**:
- Randomly selects year with weighted probabilities
- Randomly selects week (1-17)
- Recursive retry if player not found
- Uses weighted year distribution: [2017: 2.5%, 2018: 7.5%, 2019: 15%, 2020: 25%, 2021: 50%]

**Analysis**:
- ✅ Sophisticated sampling approach
- ✅ Weighted historical distribution
- ✅ Recursive retry mechanism
- ❌ **CRITICAL ISSUE**: Recursion can cause stack overflow
- ❌ **CRITICAL ISSUE**: No maximum recursion limit
- ❌ **CRITICAL ISSUE**: Can get stuck in infinite recursion

#### `get_score_for_player_safe(db, player, years, max_attempts=10)`
**Purpose**: Safe version with limited recursion
**Features**:
- Wraps original function with safety checks
- Limits maximum attempts
- Returns default score (10.0) if all attempts fail
- Handles RecursionError gracefully

**Analysis**:
- ✅ **FIXES RECURSION ISSUE**: Limited attempts prevent infinite recursion
- ✅ Graceful error handling
- ✅ Default fallback score
- ✅ Clear warning messages
- ⚠️ Default score (10.0) may not be optimal

### 4. Team Simulation

#### `simulate(team, db, years, exps=10)`
**Purpose**: Run Monte Carlo simulation for entire team
**Features**:
- Creates dataframe for simulation results
- Iterates through experiments and players
- Uses `alive_progress` for progress monitoring
- Returns simulation results dataframe

**Analysis**:
- ✅ Comprehensive team simulation
- ✅ Progress monitoring integration
- ✅ Clear result structure
- ✅ Uses safe scoring function
- ⚠️ Default 10 experiments may be too low

#### `main(years, simulations)`
**Purpose**: Main execution function
**Features**:
- Configurable years and simulation count
- Team projection calculation
- Standard deviation calculation
- Results export to TSV file

**Analysis**:
- ✅ Configurable parameters
- ✅ Statistical analysis
- ✅ Results export
- ✅ Clear output formatting

## Configuration

### Default Parameters
```python
my_years = [2017, 2018, 2019, 2020, 2021]
number_of_simulations = 5000
```

### Year Weighting
```python
p=[0.025, 0.075, 0.15, 0.25, 0.5]  # 2017-2021 weights
```

## Issues and Limitations

### Critical Issues

#### 1. Recursion Problems
**Problem**: `get_score_for_player()` can cause stack overflow
**Impact**: Script crashes with RecursionError
**Solution**: ✅ `get_score_for_player_safe()` provides fix
**Status**: Fixed with safe wrapper function

#### 2. Missing Error Handling
**Problem**: No handling for missing data or network issues
**Impact**: Script fails silently or crashes
**Solution**: Need comprehensive error handling

#### 3. Performance Issues
**Problem**: Inefficient data lookups
**Impact**: Slow execution for large datasets
**Solution**: Optimize data structures and indexing

### Minor Issues

#### 1. Hardcoded Values
**Problem**: Magic numbers and hardcoded defaults
**Impact**: Difficult to configure and maintain
**Solution**: Make configurable parameters

#### 2. Limited Validation
**Problem**: No data quality validation
**Impact**: May use poor quality data
**Solution**: Add data validation checks

## Strengths

### 1. Sophisticated Approach
- Based on established Monte Carlo methodology
- Weighted historical sampling
- Proper statistical analysis

### 2. Progress Monitoring
- Uses `alive_progress` for user feedback
- Real-time progress updates
- Professional user experience

### 3. Flexible Configuration
- Configurable years and simulation count
- Easy to modify parameters
- Clear parameter structure

### 4. Safe Implementation
- `get_score_for_player_safe()` prevents crashes
- Graceful error handling
- Default fallback values

## Integration Points

### Data Pipeline
- Reads from `datasets/` directory
- Expects combined CSV data
- Integrates with data collection pipeline

### Progress Monitoring
- Uses `alive_progress` library
- Consistent with other pipeline components
- Real-time user feedback

### Results Export
- Saves to `results/montecarlo_results/`
- TSV format for compatibility
- Date-stamped output files

## Optimization Opportunities

### 1. Data Structure Optimization
- Pre-index data by player, year, week
- Use more efficient data structures
- Cache frequently accessed data

### 2. Parallel Processing
- Parallelize simulation runs
- Use multiprocessing for large simulations
- Improve performance for high simulation counts

### 3. Memory Management
- Optimize memory usage for large datasets
- Implement data streaming for very large files
- Better memory cleanup

### 4. Error Recovery
- Implement checkpointing for long simulations
- Resume interrupted simulations
- Better error reporting and logging

## Success Criteria

### Functionality
- ✅ Monte Carlo simulation works correctly
- ✅ Team projections are calculated
- ✅ Statistical analysis is accurate
- ✅ Results are exported properly

### Performance
- ✅ Handles 5000 simulations
- ✅ Progress monitoring works
- ✅ No recursion crashes
- ⚠️ Performance could be improved

### Integration
- ✅ Reads from data pipeline
- ✅ Uses progress monitoring
- ✅ Exports results properly
- ✅ Configurable parameters

## Next Steps

### Immediate Fixes
1. **Replace recursive function**: Use iterative approach instead of recursion
2. **Add comprehensive error handling**: Handle all potential failure points
3. **Optimize data lookups**: Improve performance for large datasets

### Enhancements
1. **Add data validation**: Validate input data quality
2. **Implement parallel processing**: Speed up large simulations
3. **Add configuration management**: Make parameters more configurable
4. **Improve memory management**: Handle very large datasets efficiently

### Integration
1. **Standardize interfaces**: Consistent with other pipeline components
2. **Add logging**: Comprehensive logging for debugging
3. **Improve error reporting**: Better error messages and recovery
4. **Add unit tests**: Comprehensive test coverage

## Dependencies

### Required Packages
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `alive_progress`: Progress monitoring
- `glob`: File pattern matching
- `logging`: Logging functionality

### Data Requirements
- Combined CSV files in `datasets/` directory
- Player data with columns: Name, Position, Tm, Season, G#, FantPt
- Team file in TSV format with Name column

### Environment Requirements
- Python 3.8+
- Sufficient memory for large datasets
- Disk space for results storage
