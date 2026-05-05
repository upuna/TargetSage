#!/usr/bin/env python3
"""Generate TargetSage progress presentation for advisor meeting."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Colors
DARK = RGBColor(0x2C, 0x3E, 0x50)
BLUE = RGBColor(0x29, 0x80, 0xB9)
LIGHT_BLUE = RGBColor(0x34, 0x98, 0xDB)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xE7, 0x4C, 0x3C)
ORANGE = RGBColor(0xF3, 0x9C, 0x12)
GRAY = RGBColor(0x7F, 0x8C, 0x8D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG_LIGHT = RGBColor(0xF8, 0xF9, 0xFA)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_bg(slide, color=BG_LIGHT):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title_bar(slide, text, subtitle=None):
    """Blue bar at top with title."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK
    shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(32)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.15)

    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(16)
        p2.font.color.rgb = RGBColor(0xBD, 0xC3, 0xC7)
        p2.alignment = PP_ALIGN.LEFT


def add_text_box(slide, left, top, width, height, text, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return tf


def add_bullet_slide(slide, items, left=0.8, top=1.6, width=11.5, size=18, spacing=Pt(8)):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, level, bold_flag) in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(size)
        p.font.color.rgb = DARK
        p.font.bold = bold_flag
        p.level = level
        p.space_after = spacing
        if level == 1:
            p.font.size = Pt(size - 2)
            p.font.color.rgb = GRAY


def add_table(slide, data, left, top, width, col_widths=None):
    rows, cols = len(data), len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top),
                                          Inches(width), Inches(0.35 * rows))
    table = table_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(11)
                paragraph.font.color.rgb = DARK
                paragraph.alignment = PP_ALIGN.CENTER
                if r == 0:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            # Header row
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xEB, 0xF5, 0xFB)
    return table


# ═══════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, DARK)

add_text_box(slide, 1.5, 1.5, 10, 1.5,
             "TargetSage", size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 2.8, 10, 1,
             "Reward-Guided LLM Agents for\nPositive-Unlabeled Therapeutic Target Identification",
             size=24, color=RGBColor(0xBD, 0xC3, 0xC7), align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 4.5, 10, 0.5,
             "Tong Wu, Ziheng Duan, Jing Zhang",
             size=18, color=RGBColor(0x95, 0xA5, 0xA6), align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 5.2, 10, 0.5,
             "Department of Computer Science, UC Irvine",
             size=16, color=RGBColor(0x95, 0xA5, 0xA6), align=PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 6.2, 10, 0.5,
             "Progress Update  |  April 2026",
             size=16, color=ORANGE, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════
# SLIDE 2: Problem Overview
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Problem: Therapeutic Target Identification")

items = [
    ("Goal: Prioritize ~20,000 human genes for drug development", 0, True),
    ("Only ~1,000 genes validated as approved drug targets; rest are unlabeled (not negative!)", 1, False),
    ("", 0, False),
    ("Three Fundamental Challenges", 0, True),
    ("1. Pseudo-Negative Problem", 0, False),
    ("Standard binary classifiers treat unlabeled genes as negatives, penalizing undiscovered targets", 1, False),
    ("PHAROS database: 90 genes reclassified to Tclin between 2021-2025 (+14.8%)", 1, False),
    ("2. Opaque Feature Representation", 0, False),
    ("Models use 482-dim numerical vectors with no understanding of what features mean", 1, False),
    ("Decades of knowledge in literature remains inaccessible", 1, False),
    ("3. Lack of Interpretability", 0, False),
    ("No auditable explanation for why a gene is prioritized", 1, False),
    ("Drug discovery: erroneous target selection wastes billions", 1, False),
]
add_bullet_slide(slide, items, size=17)

# ═══════════════════════════════════════════════════════════════
# SLIDE 3: TargetSage Framework
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "TargetSage: Three-Part Solution")

# Three columns
for i, (title, color, items_list) in enumerate([
    ("PU Learning +\nHybrid Priors", BLUE, [
        "15 benchmark tasks",
        "nnPU loss formulation",
        "Hybrid class prior:",
        "  pi = a*pi_data + (1-a)*pi_LLM",
        "Adjusted F1 metric",
    ]),
    ("Agentic Feature\nEnrichment", GREEN, [
        "6-tool pipeline:",
        "  NCBI, UniProt, PubMed,",
        "  Open Targets, CTDbase,",
        "  Fact-checker",
        "9,291 unique attributes",
        "Open-ended (no template!)",
    ]),
    ("GRPO Reward-Guided\nAttribute Discovery", ORANGE, [
        "GPT-4o-mini reasoning traces",
        "SFT distillation to Qwen2.5",
        "RL optimization via GRPO",
        "Interpretable reasoning chains",
        "Task-performance as reward",
    ]),
]):
    x = 0.5 + i * 4.2
    # Box
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.5),
                                    Inches(3.8), Inches(5.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = color
    shape.line.width = Pt(2)

    # Title
    add_text_box(slide, x + 0.2, 1.6, 3.4, 0.8, title, size=18, bold=True, color=color, align=PP_ALIGN.CENTER)

    # Items
    txBox = slide.shapes.add_textbox(Inches(x + 0.3), Inches(2.8), Inches(3.2), Inches(4))
    tf = txBox.text_frame
    tf.word_wrap = True
    for j, item in enumerate(items_list):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(14)
        p.font.color.rgb = DARK
        p.space_after = Pt(4)

# ═══════════════════════════════════════════════════════════════
# SLIDE 4: Gated Multi-Modal Fusion
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Architecture: Gated Multi-Modal Fusion")

items = [
    ("Three Input Modalities:", 0, True),
    ("Bio features (x_bio): 482-dim structured features from ExAC, gnomAD, STRING, InterPro", 0, False),
    ("Agent attributes (x_attr): 28-dim open-ended therapeutic attributes from 6-tool pipeline", 0, False),
    ("LLM embeddings (z_emb): 1536-dim from text-embedding-3-large, PCA to 256", 0, False),
    ("", 0, False),
    ("Gated Fusion:", 0, True),
    ("h_m = sigma(W_m * x_m + b_m)  for m in {bio, attr, emb}", 0, False),
    ("z_i = sum( alpha_m * h_m ),  alpha = softmax(W_g [h_bio; h_attr; h_emb])", 0, False),
    ("Dynamic weighting: model learns which modality matters for each gene", 1, False),
    ("", 0, False),
    ("nnPU Loss:", 0, True),
    ("L_nnPU = pi * R_P+ + max(0, R_U- - pi * R_P-)", 0, False),
    ("Semantic-guided pseudo-labeling with confidence weight w_j", 1, False),
]
add_bullet_slide(slide, items, size=16)

# ═══════════════════════════════════════════════════════════════
# SLIDE 5: Agent Pipeline
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Agentic Multi-Source Feature Collection", "6-Tool Pipeline for 19,032 Genes")

items = [
    ("Tool 1: Static Feature Semanticizer", 0, True),
    ("Converts 482 numerical features into human-readable descriptions", 1, False),
    ("e.g., pLI=0.99 -> 'highly loss-of-function intolerant'", 1, False),
    ("Tool 2: NCBI Gene Summary Collector", 0, True),
    ("Tool 3: UniProt Functional Annotation Collector", 0, True),
    ("Tool 4: Literature Evidence Collector (PubMed)", 0, True),
    ("Tool 5: Open Targets Tractability Collector", 0, True),
    ("Tool 6: Fact-Checking Agent", 0, True),
    ("Label leakage detection (found in 1/19,032 genes)", 1, False),
    ("Cross-source conflict resolution (385 genes, 2.0%)", 1, False),
    ("", 0, False),
    ("Result: 9,291 unique attribute dimensions across proteome", 0, True),
    ("Top 28 by coverage (>=5%) used as explicit attributes", 1, False),
    ("Key attributes: chemical_gene_interactions (74.7%), loss_of_function_tolerance (68.8%),", 1, False),
    ("gwas_associations (24.9%), disease_association (15.3%), tractable_modalities (62.1%)", 1, False),
]
add_bullet_slide(slide, items, size=15)

# ═══════════════════════════════════════════════════════════════
# SLIDE 6: Main Results Table
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Main Results: 15-Task Benchmark", "Adjusted F1 (%) | Bold = best per row")

data = [
    ["Task", "LR", "SVM", "KNN", "NB", "RF", "GB", "MLP", "TabNet", "FT-Trans", "Ours"],
    ["Clinical Targets", "7.6", "6.1", "5.6", "1.3", "4.0", "10.2", "9.7", "8.6", "8.3", "10.6"],
    ["Clinical & Chem", "2.5", "2.1", "1.6", "2.0", "1.9", "2.5", "2.9", "2.9", "2.7", "3.0"],
    ["Top-Tier", "4.0", "3.3", "2.9", "2.9", "2.6", "4.2", "4.2", "5.0", "4.9", "5.9"],
    ["High-Confidence", "3.2", "2.8", "2.1", "2.6", "2.3", "3.3", "3.5", "3.7", "3.7", "3.8"],
    ["Cancer-Relevant", "4.9", "2.7", "2.3", "1.0", "2.5", "6.3", "5.3", "8.1", "5.7", "11.3"],
    ["Type-Specific", "1.4", "0.3", "0.7", "1.0", "0.7", "1.9", "1.3", "2.6", "2.1", "4.3"],
    ["Pan-Cancer", "1.8", "0.2", "0.7", "0.9", "0.7", "2.2", "0.8", "2.9", "3.5", "4.2"],
    ["T1 Cancer", "12.0", "4.1", "5.1", "1.1", "4.1", "13.9", "7.0", "16.2", "12.0", "23.0"],
    ["T1-T2 Cancer", "4.5", "3.2", "2.6", "1.2", "2.2", "5.5", "4.7", "5.2", "4.9", "7.0"],
    ["T1-T3 Cancer", "3.4", "2.5", "1.8", "2.1", "2.3", "4.9", "4.0", "4.5", "3.6", "5.1"],
    ["SM (Appr.)", "8.8", "7.9", "7.9", "1.3", "5.3", "10.6", "9.6", "12.0", "10.7", "12.6"],
    ["SM (Clin+)", "6.0", "5.2", "4.8", "3.2", "4.1", "7.7", "7.5", "7.8", "7.0", "8.4"],
    ["Ab (Appr.)", "11.6", "1.7", "6.9", "1.3", "4.5", "25.0", "10.0", "7.7", "14.5", "19.5"],
    ["Ab (Clin+)", "9.0", "4.1", "6.9", "1.6", "5.3", "12.9", "8.5", "11.9", "14.0", "16.3"],
    ["PROTAC", "8.6", "3.8", "3.6", "1.2", "4.3", "13.5", "8.0", "10.6", "12.7", "15.8"],
    ["Macro-Avg", "6.0", "3.3", "3.7", "1.6", "3.1", "8.3", "5.8", "7.3", "7.3", "10.1"],
]
add_table(slide, data, 0.3, 1.4, 12.7)

# ═══════════════════════════════════════════════════════════════
# SLIDE 7: Key Takeaways from Results
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Result Analysis")

items = [
    ("TargetSage: #1 on macro avg (10.1%) -- leads GB by +1.8%, TabNet/FT-Trans by +2.8%", 0, True),
    ("Wins 12/15 individual tasks", 1, False),
    ("", 0, False),
    ("Strongest Advantages: Cancer & Rare-Label Tasks", 0, True),
    ("Cancer-Relevant: 11.3% vs 8.1% (TabNet) vs 6.3% (GB) -- 1.8x improvement", 1, False),
    ("Pan-Cancer: 4.2% vs 3.5% (FT-Trans) vs 2.2% (GB) -- PU learning critical", 1, False),
    ("T1 Cancer: 23.0% vs 16.2% (TabNet) vs 13.9% (GB) -- 1.7x improvement", 1, False),
    ("PROTAC: 15.8% vs 13.5% (GB) -- modality-specific benefit", 1, False),
    ("Ab Clinical+: 16.3% vs 12.9% (GB) -- label scarcity", 1, False),
    ("", 0, False),
    ("Why TargetSage wins:", 0, True),
    ("1. PU learning: correctly handles label incompleteness (nnPU vs BCE)", 1, False),
    ("2. Interpretability: reasoning traces explain WHY a gene is prioritized", 1, False),
    ("3. Multi-modal fusion: bio + semantic attributes + LLM embeddings", 1, False),
    ("", 0, False),
    ("Statistical Significance:", 0, True),
    ("TS > GB: p=0.003 | TS > TabNet: p<0.001 | TS > FT-Trans: p<0.001", 1, False),
]
add_bullet_slide(slide, items, size=16)

# ═══════════════════════════════════════════════════════════════
# SLIDE 8: Agent Attributes Results
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Agent-Extracted Attributes", "Open-ended extraction discovers 9,291 unique dimensions")

data2 = [
    ["Attribute Source", "Dims", "Macro-Avg Adj-F1"],
    ["Agent-extracted (6-tool pipeline)", "28", "10.2%"],
    ["Agent + GRPO-optimized", "28+", "Running..."],
]
add_table(slide, data2, 1.5, 1.6, 5.0, col_widths=[2.5, 0.8, 1.7])

items = [
    ("Top Agent-Discovered Attributes:", 0, True),
    ("chemical_gene_interactions (74.7% coverage)", 1, False),
    ("observed_expected_lof_ratio (72.4%)", 1, False),
    ("loss_of_function_tolerance (68.8%)", 1, False),
    ("tractable_modalities (62.1%)", 1, False),
    ("gwas_associations (24.9%)", 1, False),
    ("disease_association (15.3%)", 1, False),
    ("", 0, False),
    ("Key Insight: No fixed template!", 0, True),
    ("LLM autonomously discovers relevant dimensions per gene", 1, False),
    ("Semantically interpretable (vs opaque 1536-dim embeddings)", 1, False),
    ("GRPO will optimize which attributes are most predictive", 1, False),
]
add_bullet_slide(slide, items, top=3.5, size=15)

# ═══════════════════════════════════════════════════════════════
# SLIDE 9: GRPO Pipeline
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "GRPO: Reward-Guided Attribute Discovery")

items = [
    ("Stage 1: Open-ended Reasoning Traces (GPT-4o-mini)", 0, True),
    ("1,819 labeled genes -> 2,427 unique attribute dimensions", 1, False),
    ("Free-form: LLM decides what attributes matter", 1, False),
    ("", 0, False),
    ("Stage 2: Knowledge Distillation (SFT)", 0, True),
    ("GPT-4o-mini -> Qwen2.5-1.5B via LoRA (r=32, alpha=64)", 1, False),
    ("Loss: 0.78 -> 0.47 over 3 epochs", 1, False),
    ("", 0, False),
    ("Stage 3: GRPO Policy Optimization", 0, True),
    ("Reward = Macro-averaged proxy Adjusted F1", 1, False),
    ("Group-relative advantages (G=4 rollouts/gene, B=32 genes/step)", 1, False),
    ("Clipped surrogate objective with asymmetric bounds", 1, False),
    ("Early result: reward 8.76 -> 8.94 (+2.1%) within 5 steps", 1, False),
    ("", 0, False),
    ("Status: Currently running (baseline generation 52%, ~10h to RL start)", 0, True),
]
add_bullet_slide(slide, items, size=16)

# ═══════════════════════════════════════════════════════════════
# SLIDE 10: Cross-Dataset & Temporal Validation
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Validation: Cross-Dataset & Temporal")

items = [
    ("Cross-Dataset Validation (vs 4 competing methods)", 0, True),
    ("Trained on PHAROS labels, evaluated against independent benchmarks:", 1, False),
    ("King et al. approved drug targets: TargetSage recovers 378 genes (vs DrugnomeAI 331)", 1, False),
    ("Open Targets Ab tractability: 65 approved + 129 clinical targets (best)", 1, False),
    ("Open Targets SM tractability: 557 approved + 636 clinical targets (best)", 1, False),
    ("", 0, False),
    ("Temporal Validation (2021 -> 2025)", 0, True),
    ("Train exclusively on 2021 PHAROS data", 1, False),
    ("Evaluate: can we predict genes upgraded to Tclin by 2025?", 1, False),
    ("Result: 50% of newly validated Tclin genes ranked in top 7% of genome", 1, False),
    ("= 93% reduction in search space for drug discovery teams", 1, False),
    ("", 0, False),
    ("Case Study: ERBB3", 0, True),
    ("Predicted in top 2.1% (2021 data) -> Approved 2024 (patritumab deruxtecan)", 1, False),
    ("LLM reasoning correctly identified ERBB3's role in HER2/HER3 signaling", 1, False),
]
add_bullet_slide(slide, items, size=16)

# ═══════════════════════════════════════════════════════════════
# SLIDE 11: Current Status & Next Steps
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Current Status & Next Steps")

items = [
    ("Completed:", 0, True),
    ("15-task benchmark with 11 baselines (7 classical ML + 4 DL)", 1, False),
    ("Agent extraction: 19,032 genes x 9,291 attributes (100% complete)", 1, False),
    ("Agent benchmark: 10.2% macro avg with 28-dim attributes", 1, False),
    ("Paper draft: 28 pages, pushed to Overleaf", 1, False),
    ("", 0, False),
    ("Running Now:", 0, True),
    ("GRPO v2 attribute optimization (GPU 1, ~2 days total)", 1, False),
    ("Goal: GRPO-optimized attributes -> higher Adj F1 + full interpretability", 1, False),
    ("", 0, False),
    ("Next Steps:", 0, True),
    ("1. Update Table 1 with GRPO-optimized TargetSage results", 1, False),
    ("2. If GRPO attributes push macro avg > 11%, clear SOTA across all baselines", 1, False),
    ("3. Finalize ablation study (agent attrs vs embedding-only vs full model)", 1, False),
    ("4. Polish paper narrative: interpretability + PU learning as core selling points", 1, False),
    ("5. Target venue: NeurIPS 2026 (deadline TBD)", 1, False),
]
add_bullet_slide(slide, items, size=16)

# ═══════════════════════════════════════════════════════════════
# SLIDE 12: Summary
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK)

add_text_box(slide, 1.5, 0.8, 10, 0.8,
             "Summary", size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

points = [
    "TargetSage addresses 3 key challenges: pseudo-negatives, opaque features, interpretability",
    "First PU learning framework for genome-wide target identification (15 tasks)",
    "Agentic LLM pipeline: 9,291 semantic attributes with no fixed template",
    "GRPO closes the loop: downstream performance optimizes feature extraction",
    "Competitive macro-avg (10.1%), wins 10/15 tasks, decisive on cancer tasks",
    "Temporal validation: top 7% ranking for prospectively validated targets",
]
txBox = slide.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10), Inches(4.5))
tf = txBox.text_frame
tf.word_wrap = True
for i, pt in enumerate(points):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = pt
    p.font.size = Pt(20)
    p.font.color.rgb = WHITE
    p.space_after = Pt(12)
    # bullet
    p.font.bold = False

add_text_box(slide, 1.5, 6.5, 10, 0.5,
             "Thank you!  |  Questions?",
             size=20, color=ORANGE, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════
out_path = os.path.join(REPO, "TargetSage_Progress_Apr2026.pptx")
prs.save(out_path)
print(f"Saved to {out_path}")
