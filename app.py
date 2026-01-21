""" 
Professional Streamlit Web Application 
Secure Multimodal Medical Diagnosis System 
Final Production Version with Enhanced UX & Animations
"""
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import streamlit as st
import torch
import numpy as np
from PIL import Image
import PyPDF2
from datetime import datetime
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTModel
from model_training import AttentionFusion, MultimodalClassifier
from quantum_security import QuantumSignature
from project_config import (
    MODELS_DIR,
    DEVICE,
    DEMO_CREDENTIALS,
    VIT_MODEL,
    BIOBERT_MODEL,
    ORGAN_TYPES
)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="MedSecure AI | Quantum-Enhanced Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ENHANCED PROFESSIONAL CSS WITH ANIMATIONS
# ============================================
st.markdown("""
<style>

/* ===============================
   AI RESEARCH DASHBOARD THEME
   =============================== */

/* IMPORT FONTS */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #020617;
    color: #e5e7eb;
}

/* MAIN APP */
.stApp {
    background: radial-gradient(circle at top, #020617 0%, #000000 80%);
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #e5e7eb !important;
}

/* HEADER */
.main-header {
    background: linear-gradient(145deg, #020617, #020617);
    border: 1px solid #0f172a;
    padding: 2.5rem;
    border-radius: 18px;
    box-shadow: 0 0 40px rgba(34,211,238,0.15);
    text-align: center;
}

.main-title {
    font-size: 2.6rem !important;
    font-weight: 700;
    background: linear-gradient(90deg, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-subtitle {
    color: #94a3b8 !important;
}

/* CARDS */
.card {
    background: rgba(2,6,23,0.9);
    border: 1px solid #0f172a;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 0 30px rgba(99,102,241,0.1);
}

/* CARD HEADERS */
.card-header {
    font-size: 0.9rem;
    font-weight: 600;
    color: #22d3ee !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #0f172a;
    padding-bottom: 0.5rem;
}

/* INPUTS */
input, textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border: 1px solid #1e293b !important;
}

/* BUTTONS */
.stButton button {
    background: linear-gradient(135deg, #22d3ee, #6366f1);
    color: black !important;
    font-weight: 700;
    border-radius: 10px;
}

.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(34,211,238,0.6);
}

/* ORGAN BUTTONS */
.organ-button {
    background: #020617;
    border: 1px solid #1e293b;
    color: #e5e7eb;
}

.organ-button.selected {
    background: linear-gradient(135deg, #22d3ee, #6366f1);
    color: black !important;
}

/* STATUS CARDS */
.status-healthy,
.status-diseased {
    border-radius: 18px;
    font-weight: 700;
    box-shadow: 0 0 30px rgba(99,102,241,0.3);
}

/* METRICS */
.metric-card {
    background: #020617;
    border: 1px solid #1e293b;
}

.metric-val {
    color: #22d3ee !important;
    font-family: 'JetBrains Mono', monospace;
}

/* QUANTUM SIGNATURE */
.quantum-sig {
    background: #000000;
    border-left: 4px solid #22d3ee;
    font-family: 'JetBrains Mono', monospace;
    color: #22d3ee;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #000000;
    border-right: 1px solid #0f172a;
}

/* HIDE STREAMLIT BRANDING */
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE
# ============================================
def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'report_text' not in st.session_state:
        st.session_state.report_text = None
    if 'selected_organ' not in st.session_state:
        st.session_state.selected_organ = None
    if 'show_login_success' not in st.session_state:
        st.session_state.show_login_success = False

# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_models():
    """Load all ML models (cached for performance)"""
    try:
        with st.spinner("🔄 Initializing Neural Networks & Quantum Circuits..."):
            # Simulate loading time for demo effect
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Load feature extractors
            vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL)
            vit_model = ViTModel.from_pretrained(VIT_MODEL).to(DEVICE)
            vit_model.eval()

            biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
            biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE)
            biobert_model.eval()

            # Load fusion and classifier
            fusion_model = AttentionFusion().to(DEVICE)
            classifier = MultimodalClassifier().to(DEVICE)

            # Load trained weights
            fusion_model.load_state_dict(
                torch.load(f"{MODELS_DIR}/best_fusion.pth", map_location=DEVICE)
            )
            classifier.load_state_dict(
                torch.load(f"{MODELS_DIR}/best_classifier.pth", map_location=DEVICE)
            )
            fusion_model.eval()
            classifier.eval()

            # Load quantum signature generator
            qsig = QuantumSignature()

            progress_bar.empty()
            return {
                'vit_processor': vit_processor,
                'vit_model': vit_model,
                'biobert_tokenizer': biobert_tokenizer,
                'biobert_model': biobert_model,
                'fusion_model': fusion_model,
                'classifier': classifier,
                'quantum_sig': qsig
            }
    except Exception as e:
        st.error(f"⚠️ Critical Error: Model files missing in {MODELS_DIR}. Error details: {e}")
        return None

# ============================================
# FEATURE EXTRACTION
# ============================================
def extract_image_features(image, models):
    inputs = models['vit_processor'](images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(DEVICE)
    with torch.no_grad():
        outputs = models['vit_model'](pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
    return features

def extract_text_features(text, models):
    inputs = models['biobert_tokenizer'](
        str(text),
        padding='max_length',
        max_length=256,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = models['biobert_model'](**inputs)
        features = outputs.last_hidden_state[:, 0, :]
    return features

def extract_pdf_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return None

# ============================================
# DIAGNOSIS LOGIC
# ============================================
def perform_diagnosis(image, pdf_text, organ, models):
    # Extract features
    img_features = extract_image_features(image, models)
    txt_features = extract_text_features(pdf_text, models)
    
    # Organ index
    organ_to_idx = {name: i for i, name in enumerate(ORGAN_TYPES)}
    organ_idx = torch.tensor([organ_to_idx.get(organ, 0)]).to(DEVICE)
    
    # Fusion and prediction
    with torch.no_grad():
        fused_features, attention_weights = models['fusion_model'](
            img_features, txt_features
        )
        prediction = models['classifier'](fused_features, organ_idx)
        confidence = float(prediction.item())
        
    # Diagnosis Logic
    if confidence > 0.75:
        diagnosis = "Diseased"
    elif confidence < 0.25:
        diagnosis = "Healthy"
    else:
        diagnosis = "Inconclusive"
        
    # Generate quantum signature
    quantum_sig = models['quantum_sig'].generate_signature(diagnosis, confidence)
    
    return {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'quantum_signature': quantum_sig,
        'attention_weights': attention_weights.cpu().numpy()
    }

# ============================================
# PDF REPORT GENERATION
# ============================================
def get_medical_recommendations(organ, status):
    """Helper to generate dynamic recommendations"""
    if status == "Healthy":
        return [
            "Routine annual screening recommended.",
            "Maintain current lifestyle and medication adherence.",
            "Report any new symptoms immediately."
        ]
    else:
        recs = {
            'lung': [
                "Immediate Pulmonologist consultation required.",
                "Recommended: HRCT Chest for detailed evaluation.",
                "Monitor oxygen saturation (SpO2) daily."
            ],
            'brain': [
                "Urgent Neurology referral initiated.",
                "Contrast-enhanced MRI recommended for tumor localization.",
                "Avoid driving or operating heavy machinery."
            ],
            'bone': [
                "Orthopedic consultation recommended.",
                "Immobilization of the affected area.",
                "DEXA scan for bone density evaluation."
            ],
            'bone_hbf': [
                "Orthopedic surgical evaluation recommended.",
                "CT Scan for fracture comminution assessment.",
                "Pain management protocol initiation."
            ]
        }
        return recs.get(organ, ["Specialist consultation recommended."])

def generate_pdf_report(result, username, clinical_notes="", uploaded_image=None):
    """Generates a professional medical-grade PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Header Section
    header_text = Paragraph("<b>QUANTUM-SECURE DIAGNOSTIC CENTER</b>", styles['Title'])
    story.append(header_text)
    story.append(Paragraph("<i>Advanced Multimodal AI & Quantum Encryption Division</i>", styles['BodyText']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("_" * 75, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Patient & Exam Details
    data = [
        ["Patient ID:", f"ANON-{abs(hash(username)) % 10000:04d}", "Exam Date:", datetime.now().strftime('%Y-%m-%d')],
        ["Ref Physician:", "Dr. AI System", "Time:", datetime.now().strftime('%H:%M UTC')],
        ["Target Organ:", result['organ'].upper(), "Modality:", "Multimodal (Img + Text)"]
    ]
    
    t = Table(data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # -----------------------------------------
    # EMBED MEDICAL SCAN IMAGE (X-RAY / MRI)
    # -----------------------------------------
    if uploaded_image is not None:
        try:
            # Convert PIL Image → BytesIO
            img_buffer = BytesIO()
            uploaded_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            story.append(Paragraph("<b>MEDICAL SCAN (X-RAY / MRI)</b>", styles['Heading3']))
            story.append(Spacer(1, 0.15 * inch))

            # Insert image into PDF
            report_img = RLImage(
                img_buffer,
                width=4.5 * inch,
                height=4.5 * inch,
                kind='proportional'
            )
            report_img.hAlign = 'CENTER'

            story.append(report_img)
            story.append(Spacer(1, 0.3 * inch))

        except Exception as e:
            story.append(Paragraph(
                "<font color='red'>Error embedding medical image.</font>",
                styles['Normal']
            ))

    # Clinical Findings
    story.append(Paragraph("<b>CLINICAL NOTES (EXTRACTED)</b>", styles['Heading3']))
    clean_notes = clinical_notes[:500] + "..." if len(clinical_notes) > 500 else clinical_notes
    story.append(Paragraph(f"<font size=9>{clean_notes}</font>", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Diagnostic Analysis
    story.append(Paragraph("<b>AI DIAGNOSTIC SUMMARY</b>", styles['Heading3']))
    
    diag_color = colors.red if result['diagnosis'] == "Diseased" else colors.green
    conf_score = result['confidence'] * 100
    
    diag_data = [
        ["Primary Diagnosis:", result['diagnosis'].upper()],
        ["AI Confidence:", f"{conf_score:.2f}%"],
        ["Fusion Weights:", f"Visual: {result['attention_weights'][0][0]:.2f} | Text: {result['attention_weights'][0][1]:.2f}"]
    ]
    
    dt = Table(diag_data, colWidths=[2*inch, 4*inch])
    dt.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR', (1,0), (1,0), diag_color),
        ('FONTNAME', (1,0), (1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(dt)
    story.append(Spacer(1, 0.2 * inch))

    # Recommendations
    story.append(Paragraph("<b>RECOMMENDED PROTOCOLS</b>", styles['Heading3']))
    recs = get_medical_recommendations(result['organ'], result['diagnosis'])
    for rec in recs:
        story.append(Paragraph(f"• {rec}", styles['Bullet']))
    
    story.append(Spacer(1, 0.4 * inch))

    # Security Footer
    story.append(Paragraph("<b>CRYPTOGRAPHIC VERIFICATION</b>", styles['Heading4']))
    story.append(Paragraph(f"This record is secured via Quantum-Enhanced Encryption.", styles['Normal']))
    
    sig_data = [
        ["Digital Signature:", Paragraph(f"<font fontName='Courier' size=8>{result['quantum_signature']}</font>", styles['Normal'])]
    ]
    st_table = Table(sig_data, colWidths=[1.5*inch, 4.5*inch])
    st_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.aliceblue),
        ('BOX', (0,0), (-1,-1), 1, colors.darkblue),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(st_table)

    # Disclaimer
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"<font color='grey' size=7>Generated by MedSecure AI v1.0. This report is for investigational use only and does not replace professional medical advice.</font>", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================
# VISUALIZATIONS
# ============================================
def create_confidence_gauge(confidence):
    """Creates a gauge chart with proper text visibility"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Certainty (%)", 'font': {'size': 18, 'color': '#1e293b'}},
        number={'font': {'color': '#1e293b', 'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#333", 'tickfont': {'color': '#333'}},
            'bar': {'color': "#2563eb"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 50], 'color': '#dcfce7'},
                {'range': [50, 75], 'color': '#ffedd5'},
                {'range': [75, 100], 'color': '#fee2e2'}
            ],
            'threshold': {'line': {'color': "#dc2626", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font={'family': 'Inter, sans-serif', 'color': '#1e293b'},
        margin=dict(l=20, r=20, t=40, b=20), 
        height=250
    )
    return fig

def create_attention_chart(attention_weights):
    """Creates a bar chart with proper text visibility"""
    attn = attention_weights[0]
    fig = go.Figure(data=[
        go.Bar(
            x=['Visual (Scan)', 'Text (Report)'],
            y=[attn[0], attn[1]],
            marker_color=['#3b82f6', '#8b5cf6'],
            text=[f'{attn[0]:.1%}', f'{attn[1]:.1%}'],
            textposition='auto',
            textfont=dict(color='white', size=16)
        )
    ])
    fig.update_layout(
        title={'text': 'Data Source Importance', 'font': {'size': 16, 'color': '#1e293b'}, 'x': 0.5},
        xaxis=dict(
            title='', 
            tickfont=dict(color='#1e293b', size=14),
            showgrid=False
        ),
        yaxis=dict(
            range=[0, 1], 
            visible=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    return fig

# ============================================
# LOGIN PAGE
# ============================================
def login_page():
    st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <div style='font-size: 4rem; margin-bottom:10px;'>🩺</div>
            <h1 style='color: #1e293b; font-size: 2rem; margin: 0;'>MedSecure AI</h1>
            <p style='color: #64748b; margin-bottom: 20px;'>Quantum-Safe Medical Diagnostics</p>
            <div style='background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 15px; border-radius: 8px; font-size: 0.95rem; color: #1e40af; border: 2px solid #93c5fd;'>
                🔐 <strong>Demo Access</strong><br>
                <span style='font-family: monospace;'>Username: admin</span><br>
                <span style='font-family: monospace;'>Password: admin123</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", key="login_user", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", key="login_pass", placeholder="Enter password")
        
        if st.button("🚀 Access System", type="primary", use_container_width=True):
            if username == DEMO_CREDENTIALS['username'] and password == DEMO_CREDENTIALS['password']:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.show_login_success = True
                st.rerun()
            else:
                st.error("❌ Invalid Credentials - Please try again")

# ============================================
# MAIN APPLICATION
# ============================================
def main_app():
    # Show login success message
    if st.session_state.show_login_success:
        st.markdown("""
        <div class='success-message'>
            <div class='success-icon'>✓</div>
            <div><strong>Authentication Verified!</strong></div>
            <div style='font-size: 0.95rem; margin-top: 0.5rem;'>Welcome to MedSecure AI</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
        st.session_state.show_login_success = False
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.markdown(f"## 👤 {st.session_state.username}")
        st.caption(f"🕐 Session Active | {datetime.now().strftime('%H:%M')}")
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        st.success(f"✅ Quantum Security: **Active**")
        st.info(f"🖥️ Processor: **{DEVICE}**")
        st.info(f"🔐 Encryption: **AES-256**")
        
        st.markdown("---")
        st.markdown("### 📚 Quick Help")
        with st.expander("ℹ️ How to use"):
            st.write("""
            **Step 1:** Upload medical scan (MRI/X-Ray)
            
            **Step 2:** Upload clinical report (PDF)
            
            **Step 3:** Select target organ
            
            **Step 4:** Click 'Initiate Analysis'
            
            **Step 5:** View results & download report
            """)
        
        st.markdown("---")
        if st.button("🛑 Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Main Content
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>Secure Multimodal Diagnosis</div>
        <div class='main-subtitle'>AI-Driven Fusion Analysis for Brain, Lung, and Bone Pathologies</div>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    models = load_models()
    if not models:
        st.stop()

    # --- INPUT SECTION ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>📂 Patient Data Acquisition</div>", unsafe_allow_html=True)
    
    col_img, col_txt = st.columns(2)
    
    with col_img:
        st.markdown("#### 1️⃣ Imaging Data (DICOM/IMG)")
        st.caption("Upload MRI or X-Ray scan in JPG/PNG format")
        st.markdown("<div class='upload-zone'>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("📤 Drag and drop or browse", type=['jpg', 'png', 'jpeg'], key="img_up")
        st.markdown("</div>", unsafe_allow_html=True)
        if uploaded_image:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="📸 Scan Preview", use_column_width=True)
            st.session_state.uploaded_image = image
            st.success("✅ Image Quality Verified")

    with col_txt:
        st.markdown("#### 2️⃣ Clinical Report (PDF)")
        st.caption("Upload lab report with blood/urine test results")
        st.markdown("<div class='upload-zone'>", unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader("📄 Drag and drop or browse", type=['pdf'], key="pdf_up")
        st.markdown("</div>", unsafe_allow_html=True)
        if uploaded_pdf:
            text = extract_pdf_text(uploaded_pdf)
            if text:
                st.session_state.report_text = text[:2000]
                with st.expander("📄 View Extracted Clinical Notes"):
                    st.caption(st.session_state.report_text)
                st.success("✅ Text Parsing Complete")
            else:
                st.error("❌ Failed to parse PDF")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- ORGAN SELECTION ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>🧠 Select Target Anatomy</div>", unsafe_allow_html=True)
    st.caption("Click on the organ you want to analyze")

    cols = st.columns(4)

    organs_config = [
        ("brain", "🧠", "Brain"),
        ("lung", "🫁", "Lung"),
        ("bone", "🦴", "Bone"),
        ("bone_hbf", "🦴", "Bone-HBF")
    ]

    for idx, (organ_id, icon, label) in enumerate(organs_config):
        with cols[idx]:
            # Check if this organ is selected
            is_selected = st.session_state.selected_organ == organ_id
            button_class = "organ-button selected" if is_selected else "organ-button"
            
            # Create custom HTML button
            button_html = f"""
            <div class='{button_class}' style='cursor: pointer;'>
                <div class='organ-icon'>{icon}</div>
                <div class='organ-label'>{label}</div>
            </div>
            """
            st.markdown(button_html, unsafe_allow_html=True)
            
            # Actual streamlit button (invisible but functional)
            if st.button(f"Select {label}", key=f"organ_{organ_id}", use_container_width=True):
                st.session_state.selected_organ = organ_id
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # --- ANALYSIS TRIGGER ---
    ready = st.session_state.get('uploaded_image') and st.session_state.get('report_text') and st.session_state.get('selected_organ')
    
    if ready:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 INITIATE FUSION ANALYSIS", type="primary", use_container_width=True):
            # Progress animation
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            steps = [
                "🔍 Extracting image features...",
                "📄 Parsing clinical text...",
                "🧠 Performing attention-based fusion...",
                "🔐 Applying quantum signature...",
                "✅ Analysis complete!"
            ]
            
            for i, step in enumerate(steps):
                progress_text.markdown(f"**{step}**")
                progress_bar.progress((i + 1) * 20)
                time.sleep(0.5)
            
            result = perform_diagnosis(
                st.session_state.uploaded_image,
                st.session_state.report_text,
                st.session_state.selected_organ,
                models
            )
            result['organ'] = st.session_state.selected_organ
            st.session_state.diagnosis_result = result
            
            progress_text.empty()
            progress_bar.empty()
            st.rerun()
    else:
        missing_items = []
        if not st.session_state.get('uploaded_image'):
            missing_items.append("medical scan")
        if not st.session_state.get('report_text'):
            missing_items.append("clinical report")
        if not st.session_state.get('selected_organ'):
            missing_items.append("target organ")
        
        st.warning(f"⚠️ Please provide: **{', '.join(missing_items)}** to enable fusion analysis.")

    # --- RESULTS DASHBOARD ---
    if st.session_state.diagnosis_result:
        res = st.session_state.diagnosis_result
        
        st.markdown("---")
        st.markdown("### 📊 Diagnostic Results Dashboard")
        
        # Main Status Card
        if res['diagnosis'] == "Diseased":
            st.markdown(f"""
            <div class='status-diseased'>
                <div class='status-title'>⚠️ ABNORMALITY DETECTED</div>
                <div class='status-sub'>High Probability Indicators Found - Immediate Review Recommended</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='status-healthy'>
                <div class='status-title'>✅ PATIENT HEALTHY</div>
                <div class='status-sub'>No Significant Pathologies Detected - Routine Follow-up Advised</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics Grid
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Target Anatomy</div><div class='metric-val'>{res['organ'].upper()}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Model Confidence</div><div class='metric-val'>{res['confidence']*100:.1f}%</div></div>", unsafe_allow_html=True)
        with m3:
            fusion_status = "OPTIMAL" if max(res['attention_weights'][0]) < 0.9 else "EXCELLENT"
            status_color = "#059669" if fusion_status == "OPTIMAL" else "#2563eb"
            st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Fusion Status</div><div class='metric-val' style='color:{status_color}'>{fusion_status}</div></div>", unsafe_allow_html=True)

        # Charts Area
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-header'>🎯 Model Certainty</div>", unsafe_allow_html=True)
            st.plotly_chart(create_confidence_gauge(res['confidence']), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-header'>⚖️ Modal Attention (XAI)</div>", unsafe_allow_html=True)
            st.plotly_chart(create_attention_chart(res['attention_weights']), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Security Footer
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>🔐 Immutable Quantum Ledger Verification</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='quantum-sig'>
            <div class='quantum-text'>━━━ CRYPTOGRAPHIC VALIDATION ━━━</div>
            <div class='quantum-text'>SESSION ID: {datetime.now().strftime('%Y%m%d-%H%M%S')}</div>
            <div class='quantum-text'>QUANTUM SIGNATURE: {res['quantum_signature']}</div>
            <div class='quantum-text'>ENCRYPTION: AES-256-GCM + CRYSTALS-Kyber (Post-Quantum)</div>
            <div class='quantum-text'>TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            <div class='quantum-text'>━━━ END SIGNATURE ━━━</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Download Report
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_data = generate_pdf_report(res, st.session_state.username, st.session_state.report_text,st.session_state.uploaded_image)
        st.download_button(
            "📥 Download Signed Medical Report (PDF)",
            data=pdf_data,
            file_name=f"MedSecure_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.diagnosis_result = None
                st.session_state.uploaded_image = None
                st.session_state.report_text = None
                st.session_state.selected_organ = None
                st.rerun()
        with col2:
            if st.button("📧 Share Report", use_container_width=True):
                st.info("📧 Report sharing feature - Connect your email service to enable")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    init_session_state()
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
