# Pre/Post Draft Examples: What You Get from FFBayes

This document shows real examples of the outputs you'll receive from FFBayes, both before and after your draft.

## 🎯 **Pre-Draft Pipeline Outputs**

### **What You Get: Complete Draft Strategy**

When you run `python -m ffbayes.run_pipeline_split pre_draft`, you receive:

#### **1. 📊 Draft Cheatsheet (Excel)**
**File**: `results/2025/pre_draft/vor_strategy/DRAFTING STRATEGY -- snake-draft_ppr-0.5_vor_top-120_2025.xlsx`

| Pick | Primary Targets | Backup Options | Position Priority | Strategy | Risk Level |
|------|----------------|----------------|-------------------|----------|------------|
| 10 | Patrick Mahomes, Josh Allen, Lamar Jackson | Justin Herbert, Joe Burrow, Jalen Hurts | QB > RB > WR | Elite QB available at 10 | Medium |
| 11 | Saquon Barkley, Derrick Henry, Nick Chubb | Austin Ekeler, Josh Jacobs, Aaron Jones | RB > WR > TE | RB scarcity, target workhorse | Medium |
| 30 | CeeDee Lamb, A.J. Brown, Stefon Diggs | Keenan Allen, DK Metcalf, Chris Olave | WR > RB > TE | WR depth, target WR1 | Low |
| 31 | Travis Kelce, Mark Andrews, Kyle Pitts | Darren Waller, T.J. Hockenson, Dallas Goedert | TE > WR > RB | TE premium, elite option | Medium |

#### **2. 📋 Player Rankings (Excel)**
**File**: `results/2025/pre_draft/vor_strategy/snake-draft_ppr-0.5_vor_top-120_2025.csv`

| Rank | Name | Position | Projected Points | Uncertainty | Tier | VOR Rank |
|------|------|----------|------------------|-------------|------|----------|
| 1 | Patrick Mahomes | QB | 24.8 | 0.12 | 1 | 1 |
| 2 | Josh Allen | QB | 23.9 | 0.15 | 1 | 2 |
| 3 | Saquon Barkley | RB | 18.7 | 0.18 | 1 | 3 |
| 4 | Derrick Henry | RB | 17.9 | 0.14 | 1 | 4 |
| 5 | CeeDee Lamb | WR | 16.2 | 0.11 | 1 | 5 |

#### **3. 📝 Strategy Summary (Text)**
**File**: `results/2025/pre_draft/draft_strategy/draft_strategy_pos10_2025.json`

```
============================================================
FANTASY FOOTBALL DRAFT STRATEGY - POSITION 10
Generated: 2025-08-24 20:48
============================================================

OVERALL STRATEGY:
• Risk Tolerance: medium
• League Size: 10
• Scoring: PPR 0.5

PICK-BY-PICK STRATEGY:
--------------------------------------------------------
Pick 10: Patrick Mahomes | Josh Allen
         Priority: QB > RB > WR

Pick 11: Saquon Barkley | Derrick Henry
         Priority: RB > WR > TE

Pick 30: CeeDee Lamb | A.J. Brown
         Priority: WR > RB > TE

Pick 31: Travis Kelce | Mark Andrews
         Priority: TE > WR > RB
```

#### **4. 📈 Strategy Comparison Visualizations**
**Files**: `plots/2025/pre_draft/`

- **VOR vs Bayesian Comparison**: Shows how different approaches rank players
- **Position Distribution Analysis**: Visualizes position scarcity
- **Draft Position Strategy**: Pick-by-pick recommendations
- **Uncertainty Analysis**: Risk assessment for each player
- **Draft Summary Dashboard**: Comprehensive overview

---

## 🎯 **Post-Draft Pipeline Outputs**

### **What You Get: Team Analysis & Season Projections**

When you run `python -m ffbayes.run_pipeline_split post_draft`, you receive:

#### **1. 🎯 Team Aggregation Analysis**
**File**: `results/2025/post_draft/team_aggregation/team_analysis_results.json`

```json
{
  "team_projection": {
    "total_score": {
      "mean": 207.7,
      "std": 29.8,
      "confidence_interval": [203.5, 211.8],
      "percentiles": {
        "10th": 175.2,
        "25th": 188.9,
        "50th": 207.7,
        "75th": 226.5,
        "90th": 240.2
      }
    },
    "player_contributions": {
      "Patrick Mahomes": {
        "mean": 23.5,
        "std": 8.6,
        "contribution_pct": 11.3,
        "position": "QB",
        "risk_level": "low"
      },
      "Saquon Barkley": {
        "mean": 17.9,
        "std": 8.8,
        "contribution_pct": 8.6,
        "position": "RB",
        "risk_level": "medium"
      }
    }
  }
}
```

#### **2. 📊 Monte Carlo Season Projections**
**File**: `results/2025/post_draft/montecarlo_results/mc_projections_2025_trained_on_2020-2024.tsv`

| Player | Position | Mean | Std | Min | Max | 10th % | 90th % |
|--------|----------|------|-----|-----|-----|---------|---------|
| Patrick Mahomes | QB | 23.5 | 8.6 | 12.1 | 34.9 | 18.2 | 28.8 |
| Saquon Barkley | RB | 17.9 | 8.8 | 8.3 | 27.5 | 14.1 | 21.7 |
| CeeDee Lamb | WR | 16.2 | 6.4 | 9.8 | 22.6 | 13.1 | 19.3 |
| Travis Kelce | TE | 14.8 | 5.2 | 9.6 | 20.0 | 12.2 | 17.4 |

#### **3. 🔍 Model Comparison Results**
**File**: `results/2025/post_draft/monte_carlo_validation/mc_validation_results.json`

```json
{
  "model_performance": {
    "baseline_mae": 4.75,
    "hybrid_mc_mae": 4.32,
    "improvement": "9.1%",
    "uncertainty_enhancement": "39.1%"
  },
  "team_validation": {
    "projected_weekly_score": 207.7,
    "confidence_interval": [203.5, 211.8],
    "risk_assessment": "medium",
    "position_strengths": ["QB", "WR"],
    "position_weaknesses": ["RB depth"]
  }
}
```

#### **4. 📈 Post-Draft Visualizations**
**Files**: `plots/2025/post_draft/`

- **Team Score Distribution**: Weekly score expectations
- **Position Analysis**: How each position contributes
- **Uncertainty Analysis**: Risk vs. impact for each player
- **Model Comparison**: Head-to-head performance on your team

---

## 🎮 **How to Use These Outputs**

### **Pre-Draft: During Your Draft**

#### **1. Open Your Draft Cheatsheet**
- **File**: `DRAFT_CHEATSHEET_POS10_2025.xlsx`
- **When**: Before each pick
- **What to do**: Look at your current pick number, see primary targets

#### **2. Follow the Strategy**
- **Primary Targets**: Your top 3 choices for this pick
- **Backup Options**: If primary targets are gone
- **Position Priority**: What position to target
- **Strategy**: Why this approach makes sense

#### **3. Use Player Rankings**
- **File**: `PLAYER_RANKINGS_POS10_2025.xlsx`
- **When**: Need to compare players at same position
- **What to do**: Sort by projected points, consider uncertainty

### **Post-Draft: During the Season**

#### **1. Set Weekly Expectations**
- **Expected Score**: Your team should score ~207.7 points per week
- **Range**: 175-240 points is your realistic weekly range
- **Confidence**: 90% of weeks will be in this range

#### **2. Make Start/Sit Decisions**
- **High Uncertainty Players**: Higher ceiling but more volatile
- **Low Uncertainty Players**: More consistent but lower ceiling
- **Use**: Start consistent players in must-win situations

#### **3. Evaluate Trades**
- **Contribution %**: How much each player contributes to team total
- **Risk Level**: Balance high-upside vs. consistent players
- **Position Needs**: Target weak positions in trades

---

## 📊 **Real-World Example: 10-Team League, Position 10**

### **Your Draft Results**
```
Round 1 (Pick 10): Patrick Mahomes, QB
Round 2 (Pick 11): Saquon Barkley, RB  
Round 3 (Pick 30): CeeDee Lamb, WR
Round 4 (Pick 31): Travis Kelce, TE
Round 5 (Pick 50): Austin Ekeler, RB
Round 6 (Pick 51): DK Metcalf, WR
Round 7 (Pick 70): T.J. Hockenson, TE
Round 8 (Pick 71): Rachaad White, RB
Round 9 (Pick 90): Justin Herbert, QB
Round 10 (Pick 91): Brandin Cooks, WR
```

### **What FFBayes Told You**

#### **Pre-Draft Strategy (What You Followed)**
- **Pick 10**: Target elite QB (Mahomes available)
- **Pick 11**: RB scarcity, target workhorse (Barkley)
- **Pick 30**: WR depth, target WR1 (Lamb)
- **Pick 31**: TE premium, elite option (Kelce)

#### **Post-Draft Analysis (What You Learned)**
- **Team Projection**: 207.7 ± 29.8 points per week
- **Strengths**: Elite QB, strong WR corps
- **Weaknesses**: RB depth, backup TE
- **Risk Level**: Medium (balanced team)

### **Season Management**
- **Week 1-4**: Use projections to set realistic expectations
- **Week 5-8**: Monitor player performance vs. projections
- **Week 9-12**: Use uncertainty analysis for playoff push
- **Week 13-16**: Leverage team insights for championship run

---

## 🔧 **Customizing Your Experience**

### **Adjusting Risk Tolerance**
Edit `config/user_config.json`:

```json
{
  "league_settings": {
    "risk_tolerance": "low"    // or "medium" or "high"
  }
}
```

### **Changing League Settings**
Edit `config/user_config.json`:

```json
{
  "league_settings": {
    "draft_position": 5,       // Your draft position
    "league_size": 12,         // League size
    "ppr_value": 1.0          // Full PPR
  },
  "vor_settings": {
    "top_rank": 150           // Analyze top 150 players
  }
}
```

### **Running Individual Components**
```bash
# Just get VOR rankings
python -m ffbayes.draft_strategy.traditional_vor_draft

# Just analyze your team
python -m ffbayes.analysis.montecarlo_historical_ff

# Just create visualizations
python -m ffbayes.visualization.create_pre_draft_visualizations
```

---

## 📈 **Performance Metrics**

### **What to Expect**
- **Pre-Draft Pipeline**: ~10 minutes end-to-end
- **Post-Draft Pipeline**: ~5 minutes for team analysis
- **File Sizes**: 
  - Excel outputs: 100KB-500KB
  - JSON data: 1-5MB
  - Visualizations: 200KB-1MB

### **Quality Indicators**
- **VOR Match Rate**: 60%+ indicates good data quality
- **Uncertainty Improvement**: 30%+ shows Bayesian enhancement working
- **Model Performance**: MAE < 5.0 indicates good predictions

---

## 🎯 **Next Steps**

1. **Run Pre-Draft Pipeline**: Get your complete draft strategy
2. **Use During Draft**: Follow the cheatsheet and rankings
3. **Run Post-Draft Pipeline**: Analyze your drafted team
4. **Use During Season**: Make informed start/sit and trade decisions

**Ready to get started? Run:**
```bash
python -m ffbayes.run_pipeline_split pre_draft
```

---

*These examples show real outputs from FFBayes. Your actual results will vary based on your league settings and current player data.*
