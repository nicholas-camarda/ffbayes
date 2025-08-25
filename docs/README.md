# FFBayes Documentation

Welcome to the FFBayes documentation! This guide helps you find the right information for your needs.

## 📚 **Documentation Guide**

### **🚀 New Users: Start Here**
- **[Main README](../README.md)** - Complete start-to-finish guide
- **[Pre/Post Draft Examples](PRE_POST_DRAFT_EXAMPLES.md)** - See exactly what you'll get

### **🔧 Technical Users: Deep Dive**
- **[Technical Deep Dive](TECHNICAL_DEEP_DIVE.md)** - How the models work under the hood
- **[Model Architecture](MODEL_ARCHITECTURE.md)** - Detailed model specifications
- **[API Reference](API_REFERENCE.md)** - Complete function documentation

### **📊 Data & Analysis**
- **[Data Schema](DATA_SCHEMA.md)** - Database structure and relationships
- **[Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)** - Speed and accuracy metrics
- **[Visualization Guide](VISUALIZATION_GUIDE.md)** - How to read the charts

### **🛠️ Development & Contributing**
- **[Development Setup](DEVELOPMENT_SETUP.md)** - How to set up the development environment
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project
- **[Testing Guide](TESTING_GUIDE.md)** - How to run tests and validate changes

---

## 🎯 **Quick Reference**

### **Essential Commands**
```bash
# Pre-draft strategy
python -m ffbayes.run_pipeline_split pre_draft

# Post-draft analysis  
python -m ffbayes.run_pipeline_split post_draft

# Individual components
python -m ffbayes.draft_strategy.traditional_vor_draft
python -m ffbayes.analysis.montecarlo_historical_ff
```

### **Key Files**
- **Team File**: `my_ff_teams/drafted_team_2025.tsv`
- **Main Output**: `results/2025/pre_draft/vor_strategy/DRAFTING STRATEGY -- snake-draft_ppr-0.5_vor_top-120_2025.xlsx`
- **Configuration**: JSON config files (see main README)

### **Common Issues**
- **Player Names**: Must match database exactly (e.g., "Patrick Mahomes" not "P. Mahomes")
- **File Format**: TSV with columns: POS, PLAYER, BYE
- **Environment**: Activate conda environment: `conda activate ffbayes`

---

## 📖 **Documentation Structure**

```
docs/
├── README.md                    # This file - documentation index
├── PRE_POST_DRAFT_EXAMPLES.md  # Real examples of outputs
├── TECHNICAL_DEEP_DIVE.md      # How models work under the hood
├── MODEL_ARCHITECTURE.md       # Detailed model specifications
├── API_REFERENCE.md            # Complete function documentation
├── DATA_SCHEMA.md              # Database structure
├── PERFORMANCE_BENCHMARKS.md   # Speed and accuracy metrics
├── VISUALIZATION_GUIDE.md      # How to read charts
├── DEVELOPMENT_SETUP.md        # Development environment
├── CONTRIBUTING.md             # How to contribute
└── TESTING_GUIDE.md            # Testing and validation
```

---

## 🎉 **Getting Started**

1. **Read the [Main README](../README.md)** for complete setup instructions
2. **Check [Pre/Post Draft Examples](PRE_POST_DRAFT_EXAMPLES.md)** to see what you'll get
3. **Run the pipeline**: `python -m ffbayes.run_pipeline_split pre_draft`
4. **Use the outputs** during your draft and throughout the season

---

## 🤝 **Need Help?**

- **User Issues**: Check the troubleshooting section in the main README
- **Technical Questions**: See the Technical Deep Dive documentation
- **Development**: Review the Development Setup and Contributing guides
- **Bugs**: Check the logs in the `logs/` directory

---

