import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

# Color Palette
NAVY = colors.HexColor("#0f172a")      # Dark background
BLUE = colors.HexColor("#1e3a8a")      # Corporate Blue
ACCENT = colors.HexColor("#2563eb")    # Bright Accent Blue
TEXT_DARK = colors.HexColor("#1e293b") # Primary Body Text
TEXT_LIGHT = colors.HexColor("#64748b")# Secondary/Footer Text
WHITE = colors.HexColor("#ffffff")
GRAY_BG = colors.HexColor("#f8fafc")   # Table alternating row bg
BORDER_COLOR = colors.HexColor("#e2e8f0")

class PresentationCanvas(canvas.Canvas):
    """Custom canvas to handle slide backgrounds, headers, and footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, num_pages):
        page_num = self._pageNumber
        width, height = landscape(letter)

        if page_num == 1:
            # Cover Page Background
            self.setFillColor(NAVY)
            self.rect(0, 0, width, height, fill=True, stroke=False)
            
            # Accent Stripe on Cover Page
            self.setFillColor(ACCENT)
            self.rect(0, 0, width, 12, fill=True, stroke=False)
        else:
            # Header Bar
            self.setFillColor(BLUE)
            self.rect(0, height - 50, width, 50, fill=True, stroke=False)
            
            # Header Title Text
            self.setFillColor(WHITE)
            self.setFont("Helvetica-Bold", 14)
            self.drawString(54, height - 32, "INDIA RUNS HACKATHON  |  TRACK 1: CANDIDATE DISCOVERY & RANKING")
            
            # Header Accent Line
            self.setFillColor(ACCENT)
            self.rect(0, height - 53, width, 3, fill=True, stroke=False)

            # Footer Text
            self.setFillColor(TEXT_LIGHT)
            self.setFont("Helvetica", 9)
            self.drawString(54, 30, "Intelligent Candidate Discovery & Ranking Pipeline (Senior AI Engineer)")
            self.drawRightString(width - 54, 30, f"Slide {page_num} of {num_pages}")
            
            # Footer Top Border
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(1)
            self.line(54, 45, width - 54, 45)


def build_pdf(filename="architecture_presentation.pdf"):
    # Target Landscape Letter Size
    doc = SimpleDocTemplate(
        filename,
        pagesize=landscape(letter),
        leftMargin=54,
        rightMargin=54,
        topMargin=80, # Keep space for header bar
        bottomMargin=70
    )

    styles = getSampleStyleSheet()
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=WHITE,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#94a3b8"),
        alignment=TA_CENTER
    )

    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=ACCENT,
        alignment=TA_CENTER
    )

    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=BLUE,
        spaceAfter=15
    )

    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=18,
        textColor=TEXT_DARK,
        spaceAfter=10
    )

    bullet_style = ParagraphStyle(
        'SlideBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=18,
        textColor=TEXT_DARK,
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=8,
        spaceAfter=10
    )

    story = []

    # ==================== SLIDE 1: COVER ====================
    story.append(Spacer(1, 150))
    story.append(Paragraph("INDIA RUNS HACKATHON: TRACK 1", title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("Intelligent Candidate Discovery & Ranking Pipeline", subtitle_style))
    story.append(Spacer(1, 150))
    story.append(Paragraph("DEVELOPED BY: BIPIN MAURYA  |  TRACK: AI/ML DISCOVERY  |  DATE: JUNE 2026", meta_style))
    story.append(PageBreak())

    # ==================== SLIDE 2: PROBLEM STATEMENT ====================
    story.append(Paragraph("Problem Statement & Constraints", slide_title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>The Objective:</b> Filter a pool of 100,000 resumes and identify the top 100 best-fit candidates for a <b>Senior AI Engineer</b> role at Redrob.", body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("Our architecture is strictly optimized to meet the following non-negotiable constraints:", body_style))
    
    story.append(Paragraph("• <b>Compute Limits:</b> CPU-only execution (strictly NO GPU allowed).", bullet_style))
    story.append(Paragraph("• <b>Memory Bounds:</b> Peak RAM consumption must remain under 16 GB.", bullet_style))
    story.append(Paragraph("• <b>Time Budget:</b> Execution must complete in under 5 minutes wall-clock time.", bullet_style))
    story.append(Paragraph("• <b>Network Isolation:</b> 100% offline ranking; no external API calls (e.g. OpenAI, Anthropic, Groq).", bullet_style))
    story.append(Paragraph("• <b>Intermediate Storage:</b> Under 5 GB total disk state space.", bullet_style))
    story.append(PageBreak())

    # ==================== SLIDE 3: PIPELINE ARCHITECTURE ====================
    story.append(Paragraph("Multi-Stage Hybrid Pipeline Architecture", slide_title_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("To satisfy both speed and recall constraints, we implemented a 3-Stage filtering pipeline:", body_style))
    
    # Table of Stages
    data = [
        ["Stage", "Algorithm / Model", "Purpose & Execution Details", "Output Size"],
        [
            "Stage 1: Lexical\nRetrieval",
            "BM25 Indexing",
            "Rapidly filters the 100,000 candidates down to 2,000 candidates using a highly customized positive-keyword search query based on the job description.",
            "Top 2,000"
        ],
        [
            "Stage 2: Dense\nSemantic Match",
            "Sentence Transformers\n(all-MiniLM-L6-v2)",
            "Runs locally on CPU. Encodes the 2,000 retrieved profiles into dense vectors and computes Cosine Similarity against the Job Description.",
            "Top 2,000 Reranked"
        ],
        [
            "Stage 3: Behavioral\n& Penalties",
            "Custom Signals Engine",
            "Applies targeted rules: score bonuses for ideal experience (5-9 years), Github activity, quick availability, and response rate. Traps honeypots and non-tech roles.",
            "Top 2,000 Scored"
        ]
    ]

    t = Table(data, colWidths=[110, 140, 314, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), GRAY_BG),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ('TOPPADDING', (0,1), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ==================== SLIDE 4: FUSION & TIE BREAKING ====================
    story.append(Paragraph("Rank Fusion & Format-Aligned Tie-Breaking", slide_title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Reciprocal Rank Fusion (RRF):</b>", body_style))
    story.append(Paragraph("To combine the semantic similarity scores from Stage 2 and the behavioral alignment scores from Stage 3, we fuse their ranks using RRF (constant <i>k = 60</i>) as proposed by Cormack et al. This guarantees robust and balanced integration of lexical, semantic, and resume metadata features.", bullet_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Precision-Aligned Tie-Breaking:</b>", body_style))
    story.append(Paragraph("When writing scores formatted to 6 decimal places (`:.6f`), floating point values can experience rounding collisions (e.g. at ranks 87 & 88, both rounding to `0.011188`).", bullet_style))
    story.append(Paragraph("To ensure full compatibility with the schema validator, the pipeline sorts candidates using:", bullet_style))
    
    story.append(Paragraph("key = lambda e: (-float(f\"{e['rrf_score']:.6f}\"), e['candidate_id'])", code_style))
    story.append(Paragraph("This guarantees that any identical rounded scores written to the CSV file are perfectly sorted alphabetically by <code>candidate_id</code>, satisfying the challenge's strict non-increasing rank sorting criteria.", bullet_style))
    story.append(PageBreak())

    # ==================== SLIDE 5: PERFORMANCE RESULTS ====================
    story.append(Paragraph("Performance & Verification Summary", slide_title_style))
    story.append(Spacer(1, 10))
    
    # Table of performance stats
    perf_data = [
        ["Metric / Phase", "Measured Performance", "Hackathon Limit", "Compliance Status"],
        ["Stage 1: Lexical Search (BM25)", "48.84 seconds", "N/A", "Complete"],
        ["Stage 2: Semantic Encoding", "129.45 seconds", "N/A", "Complete"],
        ["Stage 3: Behavioral Scoring", "0.29 seconds", "N/A", "Complete"],
        ["Total Execution Time (100K candidates)", "<b>184.18 seconds</b> (3.07 mins)", "5.0 minutes (300s)", "<b>PASS (38% buffer)</b>"],
        ["Memory Usage (Peak RAM)", "Well below 4 GB", "16.0 GB", "<b>PASS</b>"],
        ["Intermediate Disk Usage", "Less than 25 MB", "5.0 GB", "<b>PASS</b>"],
        ["Validation Check (`validate_submission.py`)", "<b>0 Errors (Valid Submission)</b>", "No warnings/errors", "<b>PASS</b>"]
    ]

    t_perf = Table(perf_data, colWidths=[240, 200, 124, 120])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9.5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 7),
        ('TOPPADDING', (0,1), (-1,-1), 7),
        ('BACKGROUND', (0,1), (-1,1), GRAY_BG),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor("#eff6ff")), # highlight row
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor("#f0fdf4")), # highlight pass row
    ]))
    
    story.append(t_perf)

    # Build document using the custom canvas class
    doc.build(story, canvasmaker=PresentationCanvas)

if __name__ == "__main__":
    build_pdf()
    print("PDF presentation generated successfully as architecture_presentation.pdf.")
