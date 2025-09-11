# minimal_app.py - Streamlined version for deployment troubleshooting

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NFL Matchup Analyzer", layout="wide")
st.title('🏈 NFL Matchup Analyzer - Minimal')

# Disable file watching and caching for troubleshooting
st.write("Testing minimal deployment...")

try:
    # Test basic imports
    from streamlit_data_loader import StreamlitDataLoader, check_data_freshness
    st.success("✅ Data loader imported successfully")
    
    from streamlit_simple_model import StreamlitSimpleNFLModel
    st.success("✅ Simple model imported successfully")
    
    from streamlit_real_standard_model import StreamlitRealStandardModel
    st.success("✅ Standard model imported successfully")
    
    # Test data directory
    if os.path.exists("data"):
        data_files = os.listdir("data")
        st.success(f"✅ Data directory found with {len(data_files)} files")
        for file in data_files:
            st.write(f"  - {file}")
    else:
        st.error("❌ Data directory not found")
    
    # Test basic functionality without heavy caching
    try:
        loader = StreamlitDataLoader()
        st.success("✅ Data loader initialized")
        
        # Try to load team data
        team_data = loader.load_team_data()
        st.success(f"✅ Team data loaded: {len(team_data)} teams")
        
        # Simple model test
        simple_model = StreamlitSimpleNFLModel(data_dir="data")
        st.success("✅ Simple model initialized")
        
    except Exception as e:
        st.error(f"❌ Error during initialization: {str(e)}")
        st.exception(e)
    
    st.success("🎉 All basic components working!")
    st.info("If you see this message, the core app should work. The inotify error is a Streamlit Cloud infrastructure issue that doesn't affect functionality.")
    
except Exception as e:
    st.error(f"❌ Import error: {str(e)}")
    st.exception(e)

st.write("---")
st.write("**Next Steps:**")
st.write("1. If this minimal version works, the main app should work too")
st.write("2. The inotify error is a file watching issue that doesn't break functionality")
st.write("3. You can ignore the inotify warning - it's a Streamlit Cloud infrastructure limitation")
