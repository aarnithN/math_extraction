"""Layer A (source blocks) and a stub of Layer B (interpreted segments).

COORDINATE SYSTEM (applies to every `bbox` in this package)
  * Units: PDF points (1/72 inch).
  * Space: the page's *unrotated* user space as PyMuPDF reports it for text
    extraction, origin at TOP-LEFT, y increasing downward: (x0, y0, x1, y1).
  * `PageInfo.rotation` records /Rotate. For rotation != 0 use
    `geometry.page_to_render_rect` to map into a rendered image.
  * Rendered page images use scale = dpi / 72 relative to the *rotated* page.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]


class Provenance(str, Enum):
    EXTRACTED_TEXT = "extracted_text"   # born-digital text layer
    EMBEDDED_OCR = "embedded_ocr"       # text layer that came from someone's OCR
    OCR = "ocr"                         # OCR run by this pipeline
    RULE_BASED = "rule_based"
    LLM_PROPOSED = "llm_proposed"
    HUMAN_REVIEWED = "human_reviewed"
    INFERRED = "inferred"
    USER_ANNOTATION = "user_annotation"  # highlight/comment/ink added by a PDF-software user


class PageRoute(str, Enum):
    BORN_DIGITAL = "born_digital"
    EMBEDDED_OCR_LAYER = "embedded_ocr_layer"   # ink/image page + OS OCR text
    SCANNED_NO_TEXT = "scanned_no_text"
    EMPTY = "empty"


class PageInfo(BaseModel):
    document_id: str
    page_index: int                     # zero-based
    printed_page_number: Optional[str] = None   # None = unknown
    width: float
    height: float
    rotation: int
    route: PageRoute
    route_reasons: list[str] = Field(default_factory=list)
    n_text_blocks: int
    n_image_blocks: int
    n_vector_drawings: int
    n_chars: int
    n_math_font_chars: int
    size_group: int                     # pages sharing a page size; hint of a source-document boundary
    render_path: Optional[str] = None
    render_dpi: Optional[int] = None


class SourceBlock(BaseModel):
    document_id: str
    block_id: str
    page_index: int
    printed_page_number: Optional[str] = None
    reading_order: int                  # global, 0-based, parser-native order
    block_type: Literal["text", "image"]
    raw_text: Optional[str] = None
    normalized_text: Optional[str] = None
    math_latex: Optional[str] = None    # Phase 1 never fills this
    math_latex_status: Literal["unavailable", "extracted", "ocr"] = "unavailable"
    bbox: BBox
    fonts: list[str] = Field(default_factory=list)
    max_font_size: Optional[float] = None
    baseline: Optional[float] = None        # median span origin y (page space); used to tell same-line fragments from other lines
    has_math_font: bool = False
    has_bold: bool = False
    has_italic: bool = False
    source_image_ref: Optional[str] = None   # crop path, written for flagged blocks
    crop_dpi: Optional[int] = None
    parent_section_id: Optional[str] = None  # Phase 4
    parser_name: str
    parser_version: str
    provenance: Provenance
    ocr_status: Literal["not_applicable", "embedded_unverified", "pipeline_ocr"]
    transcription_confidence: Optional[float] = None  # None = not supported
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class PageRegion(BaseModel):
    page_index: int
    bbox: BBox


class MathSymbol(BaseModel):
    """One glyph of a math expression, identified deterministically from font + Unicode + geometry."""
    char: str                       # as extracted (Unicode), may be a private-use or control char if unmapped
    code: int
    macro: Optional[str] = None     # LaTeX token, e.g. \\trianglelefteq; None if unmapped
    cls: Optional[str] = None       # greek|rel|bin|arrow|log|ord|op|open|close|punct|letter|digit|accent|name
    role: Literal["base", "sup", "sub", "num", "den", "radicand", "over", "under"] = "base"
    font: str = ""
    bbox: BBox
    mapped: bool = True
    source: Literal["table", "font_rule", "composite", "literal", "unmapped"] = "table"


class MathExpression(BaseModel):
    document_id: str
    expression_id: str
    page_index: int
    bbox: BBox
    source_block_ids: list[str]
    logical_id: Optional[str] = None
    text: str                       # characters as the parser reports them (lossy)
    latex: str                      # deterministic reconstruction
    symbols: list[MathSymbol] = Field(default_factory=list)
    structures: list[str] = Field(default_factory=list)     # scripts, fraction, sqrt, accent, long_arrow, negation ...
    unsupported: list[str] = Field(default_factory=list)    # structures we saw but could not reconstruct
    status: Literal["complete", "partial"] = "complete"
    chain: Optional[dict] = None    # arrow chains (e.g. exact sequences): nodes + labelled edges
    roundtrip: Optional[dict] = None
    provenance: Provenance = Provenance.RULE_BASED
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class FigureNode(BaseModel):
    node_id: str
    kind: Literal["text", "circle", "region", "unknown"] = "text"
    bbox: BBox
    text: str = ""
    latex: Optional[str] = None
    source_block_ids: list[str] = Field(default_factory=list)


class FigureEdge(BaseModel):
    edge_id: str
    kind: Literal["arrow", "line", "contains"] = "arrow"
    src: Optional[str] = None
    dst: Optional[str] = None
    directed: bool = True
    style: Literal["solid", "dashed", "double", "curved"] = "solid"
    head: Optional[str] = None      # arrowhead shape if any
    tail: Optional[str] = None      # e.g. "hook"
    label_text: Optional[str] = None
    label_latex: Optional[str] = None
    label_node_id: Optional[str] = None
    points: list[tuple[float, float]] = Field(default_factory=list)
    bbox: BBox
    warnings: list[str] = Field(default_factory=list)


class FigureBlock(BaseModel):
    """A diagram/plot/graph found in the vector drawing layer, kept apart from text blocks."""
    document_id: str
    figure_id: str
    page_index: int
    bbox: BBox
    kind: Literal["arrow_diagram", "graph_network", "function_plot", "region_diagram", "line_diagram",
                  "raster_image", "ink_drawing", "unclassified_drawing"]
    claimed_block_ids: list[str] = Field(default_factory=list)   # source text blocks absorbed (labels/nodes)
    nodes: list[FigureNode] = Field(default_factory=list)
    edges: list[FigureEdge] = Field(default_factory=list)
    axes: Optional[dict] = None
    curves: list[dict] = Field(default_factory=list)
    n_drawings: int = 0
    crop_ref: Optional[str] = None
    crop_dpi: Optional[int] = None
    structure_status: Literal["recovered", "partial", "not_attempted"] = "partial"
    provenance: Provenance = Provenance.RULE_BASED
    needs_review: bool = True
    warnings: list[str] = Field(default_factory=list)


class LogicalBlock(BaseModel):
    """Layer A': one or more SourceBlocks treated as a unit (e.g. an equation the parser split).
    Source blocks are never modified; merging happens only here."""
    document_id: str
    logical_id: str
    page_index: int
    member_block_ids: list[str]
    reading_order: int
    bbox: BBox                      # union of members
    raw_text: str
    normalized_text: str
    has_math_font: bool = False
    has_bold: bool = False
    max_font_size: Optional[float] = None
    provenance: Provenance
    merge_reason: Optional[str] = None
    latex_text: Optional[str] = None        # text with math zones as $...$ (deterministic reconstruction)
    expression_ids: list[str] = Field(default_factory=list)
    figure_id: Optional[str] = None         # set when this block was claimed by a figure (label/node text)
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class PdfAnnotation(BaseModel):
    """User-added annotation. Kept apart from author text; never merged into source blocks."""
    document_id: str
    annotation_id: str
    page_index: int
    subtype: str                    # Highlight, Text, FreeText, Ink, Underline, StrikeOut, Stamp, ...
    bbox: BBox
    content: str = ""               # comment / free text typed by the user
    author: Optional[str] = None
    highlighted_text: Optional[str] = None
    overlapping_block_ids: list[str] = Field(default_factory=list)
    n_ink_points: Optional[int] = None
    provenance: Provenance = Provenance.USER_ANNOTATION
    warnings: list[str] = Field(default_factory=list)


class SpanRef(BaseModel):
    logical_id: str
    start: int                      # offsets into that logical block's normalized_text
    end: int


class Reference(BaseModel):
    kind: Literal["explicit_reference", "inferred_dependency"]
    target_text: str                # e.g. "Definition 1"
    target_segment_id: Optional[str] = None
    resolved: bool = False
    candidates: list[str] = Field(default_factory=list)
    evidence_quote: str = ""


class Segment(BaseModel):
    """Layer B. Interpretation only; text and coordinates always come from the parser layer."""
    document_id: str
    segment_id: str
    segment_type: str = "unknown"
    source_block_ids: list[str]            # SourceBlock ids (parser layer)
    logical_block_ids: list[str] = Field(default_factory=list)
    source_spans: list[SpanRef] = Field(default_factory=list)   # present when quote located inside blocks
    exact_source_text: str
    page_regions: list[PageRegion]
    document_order: int
    section_title: Optional[str] = None
    source_run: Optional[int] = None       # which pasted/merged source document within the PDF
    explicit_label: Optional[str] = None
    is_explicitly_labeled: bool = False
    role_is_inferred: bool = False
    alternatives: list[str] = Field(default_factory=list)
    parent_segment_id: Optional[str] = None
    references: list[Reference] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    expression_ids: list[str] = Field(default_factory=list)
    text_latex: Optional[str] = None
    provenance: Provenance
    extraction_method: str = "rule_based"
    interpretation_status: Literal["proposed", "verified", "rejected"] = "proposed"
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)