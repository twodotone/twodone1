import streamlit as st
import pandas as pd

st.set_page_config(page_title="NFL Test App", layout="wide")
st.title('🏈 NFL Test App')

st.write("Testing basic functionality...")

try:
    from streamlit_data_loader import StreamlitDataLoader
    st.write("✅ StreamlitDataLoader imported")
    
    loader = StreamlitDataLoader()
    team_data = loader.load_team_data()
    st.write(f"✅ Team data loaded: {len(team_data)} teams")
    
    schedule_data = loader.load_schedule_data([2025])
    st.write(f"✅ Schedule data loaded: {len(schedule_data)} games")
    
except Exception as e:
    st.error(f"❌ Data loader error: {e}")
    import traceback
    st.code(traceback.format_exc())

try:
    from streamlit_simple_model import StreamlitSimpleNFLModel
    st.write("✅ StreamlitSimpleNFLModel imported")
    
    simple_model = StreamlitSimpleNFLModel(data_dir="data")
    simple_model.load_data_from_parquet([2022, 2023, 2024])
    st.write("✅ Simple model data loaded")
    
except Exception as e:
    st.error(f"❌ Simple model error: {e}")
    import traceback
    st.code(traceback.format_exc())

try:
    from streamlit_dynamic_model import StreamlitDynamicSeasonModel
    st.write("✅ StreamlitDynamicSeasonModel imported")
    
except Exception as e:
    st.error(f"❌ Dynamic model error: {e}")
    import traceback
    st.code(traceback.format_exc())

st.write("Test complete!")
