import streamlit as st

def render_footer():
    """
    Renders a premium, compact, and high-visibility footer with social links.
    """
    st.markdown("---")
    
    # Custom CSS for a more compact and visible footer
    footer_style = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .footer-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1.5rem 0;
            background: transparent;
            color: #212529; /* High contrast text */
            margin-top: 1rem;
            width: 100%;
        }
        
        .social-icons {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 0.75rem;
        }
        
        .social-icon {
            font-size: 1.2rem;
            color: #495057;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        
        .social-icon:hover {
            color: #ff4b4b;
            transform: scale(1.1);
        }
        
        .footer-text {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
            color: #343a40;
        }
        
        .footer-subtext {
            font-size: 0.75rem;
            color: #495057;
        }
        
        .mail-link {
            color: #0056b3;
            text-decoration: none;
            font-weight: 700;
        }
        
        .mail-link:hover {
            text-decoration: underline;
        }
    </style>
    
    <div class="footer-container">
        <div class="social-icons">
            <a href="https://khanz9664.github.io/portfolio" target="_blank" class="social-icon" title="Portfolio">
                <i class="fas fa-globe"></i>
            </a>
            <a href="https://github.com/khanz9664" target="_blank" class="social-icon" title="GitHub">
                <i class="fab fa-github"></i>
            </a>
            <a href="https://www.linkedin.com/in/shahid-ul-islam-13650998/" target="_blank" class="social-icon" title="LinkedIn">
                <i class="fab fa-linkedin"></i>
            </a>
            <a href="https://www.kaggle.com/shaddy9664" target="_blank" class="social-icon" title="Kaggle">
                <i class="fab fa-kaggle"></i>
            </a>
            <a href="https://instagram.com/shaddy9664" target="_blank" class="social-icon" title="Instagram">
                <i class="fab fa-instagram"></i>
            </a>
        </div>
        <div class="footer-text">
            Developed by <span style="color: #000000; font-weight: 800;">Shahid Ul Islam</span>
        </div>
        <div class="footer-subtext">
            Contact: <a href="mailto:shahid9664@gmail.com" class="mail-link">shahid9664@gmail.com</a>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.65rem; color: #868e96; letter-spacing: 0.5px; text-transform: uppercase;">
            © 2026 CardioSense AI
        </div>
    </div>
    """
    
    st.markdown(footer_style, unsafe_allow_html=True)
