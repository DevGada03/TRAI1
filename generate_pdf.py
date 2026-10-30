import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak, Table, TableStyle
)
from reportlab.pdfgen import canvas

base_dir = r"c:\Users\DevG\Desktop\TRAI1"
pdf_path = os.path.join(base_dir, "PrivateMed_Website_Code.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#6E6A63"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 26, "PrivateMed — Clinical AI Privacy Lab · Full Source Code Documentation")
            self.setStrokeColor(colors.HexColor("#E2DDD5"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 22, page_text)
        self.drawString(36, 22, "CONFIDENTIAL · CLINICAL DATA PROTECTION & TECHNOLOGY")
        self.setStrokeColor(colors.HexColor("#E2DDD5"))
        self.setLineWidth(0.5)
        self.line(36, 30, 8.5 * inch - 36, 30)
        self.restoreState()

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1E1E1C"),
    spaceAfter=6
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#6E6A63"),
    spaceAfter=14
)

section_heading = ParagraphStyle(
    "SectionHeading",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    textColor=colors.HexColor("#C85A32"),
    spaceBefore=12,
    spaceAfter=4,
    keepWithNext=True
)

file_desc_style = ParagraphStyle(
    "FileDesc",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor("#3D6B58"),
    spaceAfter=8
)

code_style = ParagraphStyle(
    "CodeBlock",
    fontName="Courier",
    fontSize=6.8,
    leading=8.8,
    textColor=colors.HexColor("#1E1E1C")
)

files_to_include = [
    (
        "1. Website Frontend (HTML, CSS & JavaScript)",
        os.path.join(base_dir, "static", "index.html"),
        "Human-crafted editorial dashboard: patient memorization visualizer, interactive training lab, and Chart.js dual plots."
    ),
    (
        "2. FastAPI Web Server & API Router",
        os.path.join(base_dir, "api", "app.py"),
        "Backend handling /api/train, /api/status, /api/sweep, and /api/laplace/simulate endpoints."
    ),
    (
        "3. Core DP-SGD Training Engine",
        os.path.join(base_dir, "training", "dp_trainer.py"),
        "PyTorch training loop integrating Opacus PrivacyEngine with Renyi DP (RDP) accounting and Epsilon Guardrail."
    ),
    (
        "4. Sensitivity & Laplace Mechanism Module",
        os.path.join(base_dir, "training", "mechanisms.py"),
        "Global L1 sensitivity computation across 16 EHR features and sequential privacy budget composition."
    ),
    (
        "5. Clinical Neural Network Architecture",
        os.path.join(base_dir, "models", "clinical_net.py"),
        "Opacus-compliant multi-layer perceptron (ClinicalMLP) with LayerNorm to prevent cross-sample batch leakage."
    ),
    (
        "6. Membership Inference Attack (MIA) Auditor",
        os.path.join(base_dir, "evaluation", "mia_audit.py"),
        "Empirical privacy leakage assessment comparing training member vs test holdout loss distributions."
    ),
    (
        "7. Clinical Synthetic EHR Generator",
        os.path.join(base_dir, "data", "clinical_generator.py"),
        "Generates 5,000 synthetic patient records across 16 clinical markers with realistic physiological correlations."
    ),
    (
        "8. Verification Pipeline Script",
        os.path.join(base_dir, "test_pipeline.py"),
        "Automated self-test script verifying gradient clipping, epsilon consumption bounds, and MIA protection."
    ),
]

elements = []

# Title Section
elements.append(Paragraph("PrivateMed: Complete Website & Backend Codebase", title_style))
elements.append(Paragraph(
    "Source code documentation for <b>PrivateMed: Differentially Private Clinical Model Training under Epsilon Budget</b>.<br/>"
    "<b>Architecture:</b> FastAPI Server · Opacus DP-SGD · PyTorch · LayerNorm MLP · Chart.js · Editorial UI",
    subtitle_style
))

# Summary Table
table_data = [
    ["#", "Module / File", "Primary Role", "Lines"]
]
for idx, (title, path, desc) in enumerate(files_to_include, 1):
    try:
        with open(path, "r", encoding="utf-8") as f:
            line_count = len(f.readlines())
    except:
        line_count = "N/A"
    fname = os.path.relpath(path, base_dir)
    table_data.append([str(idx), fname, desc[:58] + "...", str(line_count)])

t = Table(table_data, colWidths=[24, 150, 310, 46])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F7F5F0")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#1E1E1C")),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E2DDD5")),
    ("ALIGN", (0,0), (0,-1), "CENTER"),
    ("ALIGN", (3,0), (3,-1), "RIGHT"),
]))
elements.append(t)
elements.append(Spacer(1, 16))
elements.append(PageBreak())

# Add Code Sections
for title, path, desc in files_to_include:
    elements.append(Paragraph(title, section_heading))
    rel_path = os.path.relpath(path, base_dir)
    elements.append(Paragraph(f"<b>Path:</b> {rel_path} &nbsp;|&nbsp; <i>{desc}</i>", file_desc_style))
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            code_text = f.read()
        
        lines = code_text.splitlines()
        chunk_size = 75
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i+chunk_size])
            clean_chunk = (
                chunk.replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;")
            )
            elements.append(Preformatted(clean_chunk, code_style))
    else:
        elements.append(Paragraph("[File not found]", file_desc_style))
    
    elements.append(Spacer(1, 10))
    elements.append(PageBreak())

# Remove trailing pagebreak
if elements and isinstance(elements[-1], PageBreak):
    elements.pop()

doc.build(elements, canvasmaker=NumberedCanvas)
print(f"SUCCESS: Generated {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
