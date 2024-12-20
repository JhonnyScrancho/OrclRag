import streamlit as st

def apply_custom_styles():
    """Apply custom styles to the Streamlit app."""
    st.markdown("""
        <style>
            /* Main container styles */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Sidebar styles */
            .css-1d391kg {
                background-color: #f8f9fa;
            }
            
            /* Logo styles */
            .logo-title {
                text-align: center;
                font-size: 2em !important;
                font-weight: bold !important;
            }
            
            .img {
                border-radius: 60% !important;
            }
            
            /* Custom tab styles */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
                padding: 0.5rem 1rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 4px;
                color: #0e1117;
                padding: 0.5rem 1rem;
            }
            
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                background-color: #1f77b4;
                color: white;
            }
            
            /* Chat container styles */
            .chat-container {
                background-color: white;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* File uploader styles */
            .uploadedFile {
                border: 2px dashed #1f77b4;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                margin: 1rem 0;
            }
            
            /* Custom button styles */
            .stButton>button {
                width: 100%;
                border-radius: 5px;
                background-color: #1f77b4;
                color: white;
            }
            
            /* Custom metric styles */
            .css-1xarl3l {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 5px;
            }
            
            /* Chat input styles */
            .stChatInputContainer {
                padding: 1rem;
                background-color: #f8f9fa;
                border-radius: 10px;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with logo and navigation."""
    with st.sidebar:
        # Logo con bordo circolare
        st.image("src/img/logo.png", use_column_width=True)
        
        # Titolo sotto il logo
        st.markdown('<h1 class="logo-title">L\'Oracolo</h1>', unsafe_allow_html=True)
        
        # Navigation menu
        st.markdown("---")
        selected = st.radio(
            "",
            ["💬 Chat", "📊 Database", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        # File uploader section in sidebar
        st.markdown("---")
        st.markdown("### Upload Forum JSON")
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=['json'],
            help="Limit 200MB per file • JSON"
        )
        
        return selected, uploaded_file