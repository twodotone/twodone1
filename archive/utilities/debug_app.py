import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NFL Debug App", layout="wide")
st.title('🏈 NFL App Debug Test')

st.write("Testing basic functionality...")

# Test 1: Basic Streamlit
st.success("✅ Streamlit is working")

# Test 2: Data directory
if os.path.exists("data"):
    st.success("✅ Data directory exists")
    data_files = [f for f in os.listdir("data") if f.endswith('.parquet')]
    st.write(f"Data files found: {len(data_files)}")
    for file in data_files[:5]:  # Show first 5 files
        st.write(f"  • {file}")
else:
    st.error("❌ Data directory not found")

# Test 3: Try importing our modules
try:
    from streamlit_data_loader import StreamlitDataLoader
    st.success("✅ StreamlitDataLoader imported")
    
    loader = StreamlitDataLoader()
    team_data = loader.load_team_data()
    st.success(f"✅ Team data loaded: {len(team_data)} teams")
    
except Exception as e:
    st.error(f"❌ Data loader error: {e}")
    import traceback
    st.code(traceback.format_exc())

# Test 4: Try Simple Model
try:
    from streamlit_simple_model import StreamlitSimpleNFLModel
    st.success("✅ SimpleModel imported")
    
except Exception as e:
    st.error(f"❌ Simple model error: {e}")
    import traceback
    st.code(traceback.format_exc())

# Test 5: Try Standard Model  
try:
    from streamlit_real_standard_model import StreamlitRealStandardModel
    st.success("✅ StandardModel imported")
    
except Exception as e:
    st.error(f"❌ Standard model error: {e}")
    import traceback
    st.code(traceback.format_exc())

st.write("Debug test complete!")
