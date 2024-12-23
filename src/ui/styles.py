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
                padding: 0 !important;
                margin: 0 !important;
            }
            
            img {
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
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
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
                background: linear-gradient(
                    45deg,
                    #FF69B4, /* Hot pink */
                    #DA70D6, /* Orchid */
                    #87CEEB, /* Sky blue */
                    #DDA0DD, /* Plum */
                    #98FB98, /* Pale green */
                    #FFB6C1  /* Light pink */
                );
                background-size: 300% 300%;
                animation: rainbow 12s ease infinite;
                color: white;
                border: none;
                transition: all 0.3s ease;
            }

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

            /* Hover effect */
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 0 15px rgba(255, 182, 193, 0.5);
                background: linear-gradient(
                    45deg,
                    #FFB6C1,
                    #98FB98,
                    #DDA0DD,
                    #87CEEB,
                    #DA70D6,
                    #FF69B4
                );
            }

            /* Active effect */
            .stButton>button:active {
                background: white;
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
                transform: translateY(1px);
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
    # Inizializza la session state se non esiste
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "💬 Chat"
        
    def nav_to(page):
        st.session_state.current_page = page
        st.rerun()
    
    # Logo con bordo circolare
    st.sidebar.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.sidebar.image("src/img/logo.png", use_column_width=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Titolo sotto il logo
    st.sidebar.markdown('<h1 class="logo-title">L\'Oracolo</h1>', unsafe_allow_html=True)
    
    # Navigation menu
    st.sidebar.markdown("---")
    
    # Navigation buttons with active state
    for page in ["💬 Chat", "📊 Database", "⚙️ Settings"]:
        button_style = "active" if st.session_state.current_page == page else ""
        st.sidebar.markdown(f'''
            <style>
                div[data-testid="stHorizontalBlock"] button[kind="{page}"] {{
                    background: {f"white !important" if button_style == "active" else "var(--primary-color)"};
                    color: {f"black !important" if button_style == "active" else "white"};
                    border: {f"2px solid var(--primary-color) !important" if button_style == "active" else "none"};
                }}
            </style>
        ''', unsafe_allow_html=True)
        
        if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True, type="primary", kwargs={"kind": page}):
            nav_to(page)
    
    # File uploader section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Carica JSON")
    uploaded_file = st.sidebar.file_uploader(
        "",
        type=['json'],
        help="Limit 200MB per file • JSON"
    )
    
    # Add container for process button at bottom
    st.markdown("""
        <div class="process-file-container">
            <div id="process-button-placeholder"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Stili aggiuntivi per sidebar e componenti
    st.sidebar.markdown("""
        <style>
        /* Stili container immagine */
        .img-container img {
            border-radius: 50%;
            border: 2px solid #ffffff;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Stili titolo logo */
        .logo-title {
            text-align: center;
            margin-top: 1rem;
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(45deg, #FF69B4, #DA70D6, #87CEEB);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 0.5rem 0;
        }
        
        /* Stili bottoni navigazione */
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            transition: all 0.3s ease;
            text-align: left;
            font-size: 1rem;
            padding: 0.5rem 1rem;
            margin-bottom: 0.5rem;
            border: none;
            background: linear-gradient(
                45deg,
                #FF69B4,
                #DA70D6,
                #87CEEB
            );
            background-size: 200% 200%;
            animation: gradient 5s ease infinite;
            color: white;
            font-weight: 500;
        }
        
        /* Animazione gradiente */
        @keyframes gradient {
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
        
        /* Effetti hover bottoni */
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            background-position: right center;
        }
        
        /* Effetti click bottoni */
        .stButton>button:active {
            transform: translateY(1px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Stili divisori */
        hr {
            margin: 1.5rem 0;
            border: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, #e0e0e0, transparent);
        }
        
        /* Stili uploader file */
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
        
        /* Stili container processo file */
        .process-file-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 17rem;
            background-color: white;
            padding: 1rem;
            border-top: 1px solid #e6e6e6;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            z-index: 1000;
        }
        
        /* Stili sidebar principale */
        .css-1d391kg {
            background-color: #f8f9fa;
            padding-bottom: 5rem !important;  /* Spazio per il bottone fisso */
        }
        
        /* Stili header sezioni */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .process-file-container {
                width: 100%;
                left: 0;
            }
            
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
    
    return st.session_state.current_page, uploaded_file