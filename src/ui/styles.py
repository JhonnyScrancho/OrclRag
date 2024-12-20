import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the application"""
    st.set_page_config(
        page_title="🔮 L'Oracolo",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        /* Logo and title styling */
        img {
            border-radius: 60% !important;
        }
        
        .logo-title {
            text-align: center;
            font-size: 2em !important;
            font-weight: bold !important;
        }
        
        /* Modern color scheme */
        :root {
            --primary-color: #7C3AED;
            --secondary-color: #4F46E5;
            --background-color: #F9FAFB;
            --text-color: #1F2937;
            --sidebar-color: #F3F4F6;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--sidebar-color);
            padding: 2rem 1rem;
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Chat container */
        .chat-container {
            height: calc(100vh - 280px);
            overflow-y: auto;
            padding: 1rem;
            margin-bottom: 80px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Fixed chat input at bottom */
        .stChatInputContainer {
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
            max-width: 800px;
            background: white;
            padding: 1rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        
        /* Modern button styling */
        .stButton>button {
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton>button:hover {
            background: var(--secondary-color);
            transform: translateY(-1px);
        }
        
        /* Card styling */
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Message bubbles */
        .message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            max-width: 80%;
        }
        
        .user-message {
            background: var(--primary-color);
            color: white;
            margin-left: auto;
        }
        
        .assistant-message {
            background: var(--sidebar-color);
            color: var(--text-color);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: nowrap;
            border-radius: 6px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .chat-container {
                height: calc(100vh - 200px);
            }
            
            .message {
                max-width: 90%;
            }
        }
    </style>
    """, unsafe_allow_html=True)