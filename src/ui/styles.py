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
                padding-bottom: 1rem;
            }
            
            .sidebar .sidebar-content {
                background-color: #f8f9fa;
            }
            
            /* Logo and title styles */
            .logo-title {
                text-align: center;
                font-size: 2em !important;
                font-weight: bold !important;
                padding: 0 !important;
                margin: 0 !important;
                background: linear-gradient(45deg, #FF69B4, #DA70D6, #87CEEB);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 0.5rem 0;
            }
            
            img {
                border-radius: 50% !important;
                border: 2px solid #ffffff;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            /* Tab styles */
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
            
            /* Chat styles */
            .chat-container {
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
            
            .stChatInputContainer {
                padding: 1rem;
                background-color: #f8f9fa;
                border-radius: 10px;
                margin-top: 1rem;
            }
            
            /* File uploader styles */
            .uploadedFile {
                border: 2px dashed #1f77b4;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                transition: all 0.3s ease;
            }
            
            .uploadedFile:hover {
                border-color: #DA70D6;
                background-color: rgba(218,112,214,0.05);
            }
            
            /* Button styles */
            .stButton>button {
                width: 100%;
                border-radius: 5px;
                background: linear-gradient(
                    45deg,
                    #FF69B4,
                    #DA70D6,
                    #87CEEB,
                    #DDA0DD,
                    #98FB98,
                    #FFB6C1
                );
                background-size: 300% 300%;
                animation: rainbow 12s ease infinite;
                color: white;
                border: none;
                transition: all 0.3s ease;
                text-align: left;
                font-size: 1rem;
                padding: 0.5rem 1rem;
                margin-bottom: 0.5rem;
                font-weight: 500;
            }
            
            /* Button hover effect */
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                background-position: right center;
            }
            
            /* Button active effect */
            .stButton>button:active {
                transform: translateY(1px);
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                border: 3px solid transparent;
                background-image: linear-gradient(white, white), 
                linear-gradient(
                    45deg,
                    #FF69B4,
                    #DA70D6,
                    #87CEEB,
                    #DDA0DD,
                    #98FB98,
                    #FFB6C1
                );
                background-origin: border-box;
                background-clip: content-box, border-box;
                color: #333 !important;
            }
            
            /* Metric styles */
            .css-1xarl3l {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 5px;
            }
            
            /* Divider styles */
            hr {
                margin: 1.5rem 0;
                border: 0;
                height: 1px;
                background: linear-gradient(to right, transparent, #e0e0e0, transparent);
            }

            /* Agent control section */
            .agent-controls {
                background-color: #ffffff;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .agent-controls h3 {
                color: #1f77b4;
                font-size: 1.2em;
                margin-bottom: 1rem;
            }
            
            .token-info {
                background-color: #e9ecef;
                padding: 0.5rem;
                border-radius: 5px;
                margin-top: 0.5rem;
                font-size: 0.9em;
            }
            
            /* Animation keyframes */
            @keyframes rainbow {
                0% {
                    background-position: 0% 50%;
                }
                50% {
                    background-position: 100% 50%;
                }
                100% {
                    background-position: 0% 50%;
                }
            }
            
            /* Responsive styles */
            @media (max-width: 768px) {
                .stButton>button {
                    font-size: 0.9rem;
                    padding: 0.4rem 0.8rem;
                }
                
                .logo-title {
                    font-size: 1.5em;
                }
            }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with logo, navigation, and agent controls."""
    # Initialize session state if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "💬 Chat"
    if 'num_agents' not in st.session_state:
        st.session_state.num_agents = 3
    if 'show_agent_details' not in st.session_state:
        st.session_state.show_agent_details = False
        
    def nav_to(page):
        st.session_state.current_page = page
        st.rerun()
    
    # Logo with circular border
    st.sidebar.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.sidebar.image("src/img/logo.png", use_column_width=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Title under logo
    st.sidebar.markdown('<h1 class="logo-title">L\'Oracolo</h1>', unsafe_allow_html=True)
    
    # Agent Controls Section
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="agent-controls">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🤖 Controllo Agenti")
    
    # Number of agents slider
    num_agents = st.sidebar.slider(
        "Numero di agenti analisti",
        min_value=3,
        max_value=10,
        value=st.session_state.num_agents,
        step=1,
        help="Aumenta il numero di agenti se l'analisi va in overflow di token"
    )
    if num_agents != st.session_state.num_agents:
        st.session_state.num_agents = num_agents
    
    # Show agent details toggle
    show_details = st.sidebar.toggle(
        "Mostra dettagli analisi",
        value=st.session_state.show_agent_details,
        help="Visualizza l'output di ogni singolo agente"
    )
    if show_details != st.session_state.show_agent_details:
        st.session_state.show_agent_details = show_details
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation menu
    st.sidebar.markdown("---")
    
    # Navigation buttons with active state
    for page in ["💬 Chat", "📊 Database", "⚙️ Settings"]:
        button_style = "active" if st.session_state.current_page == page else ""
        st.sidebar.markdown(f'''
            <style>
                div[data-testid="stHorizontalBlock"] button[kind="{page}"] {{
                    background: {f"var(--primary-color)" if button_style == "active" else "var(--primary-color)"};
                    color: {f"black !important" if button_style == "active" else "white"};
                    border: {f"2px solid var(--primary-color) !important" if button_style == "active" else "none"};
                }}
            </style>
        ''', unsafe_allow_html=True)
        
        if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True):
            nav_to(page)
    
    # File uploader section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Carica JSON")
    uploaded_file = st.sidebar.file_uploader(
        "",
        type=['json'],
        help="Limit 200MB per file • JSON"
    )
    
    return st.session_state.current_page, uploaded_file