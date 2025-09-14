# NFL Matchup Analyzer - Streamlined Project Structure

## 🎯 **PRODUCTION FILES (Core App)**

### **Main Application:**
- `app_streamlit.py` - Production Streamlit app with both models

### **Model Components:**
- `streamlit_simple_model.py` - EPA-based simple model
- `streamlit_real_standard_model.py` - Complex tiered historical stats model  
- `streamlit_data_loader.py` - Data loading and freshness checking

### **Backend Dependencies:**
- `stats_calculator.py` - Core statistical functions
- `data_loader.py` - Rolling data loading utilities
- `team_mapping.py` - Team abbreviation mappings
- `dynamic_hfa.py` - Dynamic home field advantage
- `hfa_data.py` - Home field advantage data

### **Data & Configuration:**
- `data/` - All parquet data files
- `.streamlit/config.toml` - Streamlit configuration
- `requirements_streamlit.txt` - Production dependencies
- `.github/workflows/` - Automated data updates

### **Update Scripts:**
- `update_data_for_streamlit.py` - Data update automation
- `download_data.py` - Fresh data downloading
- `update_data.py` - Legacy data updater
- `update_lines.py` - Betting lines updater

## 📂 **ARCHIVED FILES**

### **Old App Versions (`archive/old_apps/`):**
- `app.py` - Original full-featured app
- `app_beta.py` - Beta version
- `app_simple.py` - Simplified version
- `app_streamlit_safe.py` - Safe mode version
- `minimal_app.py` - Minimal test version
- `test_app.py` - Basic test app

### **Research & Analysis (`archive/research/`):**
- `*backtest*.py` - Backtesting scripts
- `*analysis*.py` - Data analysis scripts  
- `*comparison*.py` - Model comparison tools
- `confidence_ratings.py` - Confidence analysis
- `variance_model.py` - Variance modeling
- `weight_optimizer.py` - Weight optimization
- `value_*.py` - Value betting analysis

### **Experimental Models (`archive/experiments/`):**
- `dynamic_season_model.py` - Dynamic modeling experiment
- `season_ready_model.py` - Season-specific model
- `streamlit_dynamic_model.py` - Dynamic Streamlit model
- `*demo*.py` - Demo scripts

### **Utilities & Debug (`archive/utilities/`):**
- `debug_*.py` - Debug tools
- `test_*.py` - Test scripts
- `verify_*.py` - Verification utilities
- `deployment_checklist.py` - Deployment helper
- `*.txt` - Documentation files
- `*.csv`, `*.png` - Analysis outputs

## 🚀 **To Run the App:**

```bash
# Local development
streamlit run app_streamlit.py

# Update data
python update_data_for_streamlit.py --update-2025-stats
```

## 📊 **Models Available:**

1. **Simple Model**: Pure EPA-based predictions
2. **Standard Model**: Complex tiered historical stats with SOS adjustments

## 🔄 **Automated Updates:**

- **Daily 6 AM EST**: Betting lines update
- **Tuesday 8 AM EST**: 2025 season stats update
- **Manual trigger**: Available via GitHub Actions

---

*Project cleaned up on September 14, 2025 - Streamlined from 60+ files to 15 core files*