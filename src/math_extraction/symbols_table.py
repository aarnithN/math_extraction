"""Unicode -> LaTeX macro tables and font-family rules for deterministic math-symbol recovery.

Two kinds of knowledge live here:
  * UNICODE_MACROS: what a Unicode math character is in LaTeX (class follows TeX atom classes).
  * Font-aware exceptions: TeX fonts (CMSY, CMMI, CMEX, MSBM, ...) whose glyphs PyMuPDF reports with the
    wrong or an ambiguous Unicode value (found by inspecting real PDFs). Unknown glyphs are NEVER guessed:
    they are returned unmapped so a later OCR/VLM pass can double-check them.
"""
from __future__ import annotations

import re

_RAW = r"""
α \alpha greek|β \beta greek|γ \gamma greek|δ \delta greek|ϵ \epsilon greek|ε \varepsilon greek|ζ \zeta greek|η \eta greek
θ \theta greek|ϑ \vartheta greek|ι \iota greek|κ \kappa greek|ϰ \varkappa greek|λ \lambda greek|μ \mu greek|ν \nu greek
ξ \xi greek|π \pi greek|ϖ \varpi greek|ρ \rho greek|ϱ \varrho greek|σ \sigma greek|ς \varsigma greek|τ \tau greek
υ \upsilon greek|ϕ \phi greek|φ \varphi greek|χ \chi greek|ψ \psi greek|ω \omega greek
Γ \Gamma greek|Δ \Delta greek|Θ \Theta greek|Λ \Lambda greek|Ξ \Xi greek|Π \Pi greek|Σ \Sigma greek|Υ \Upsilon greek
Φ \Phi greek|Ψ \Psi greek|Ω \Omega greek
∈ \in rel|∉ \notin rel|∋ \ni rel|⊂ \subset rel|⊃ \supset rel|⊆ \subseteq rel|⊇ \supseteq rel|⊄ \not\subset rel
⊈ \nsubseteq rel|⊊ \subsetneq rel|⊋ \supsetneq rel|≤ \leq rel|≥ \geq rel|≠ \neq rel|≈ \approx rel|≅ \cong rel
≡ \equiv rel|∼ \sim rel|≃ \simeq rel|∝ \propto rel|≪ \ll rel|≫ \gg rel|≺ \prec rel|≻ \succ rel|≼ \preceq rel|≽ \succeq rel
⊴ \trianglelefteq rel|⊵ \trianglerighteq rel|◁ \lhd bin|▷ \rhd bin|⊲ \vartriangleleft rel|⊳ \vartriangleright rel
∣ \mid rel|∤ \nmid rel|∥ \parallel rel|⊥ \perp rel|⊢ \vdash rel|⊣ \dashv rel|⊨ \models rel|≍ \asymp rel|≐ \doteq rel
⊏ \sqsubset rel|⊑ \sqsubseteq rel|⊐ \sqsupset rel|⊒ \sqsupseteq rel|≲ \lesssim rel|≳ \gtrsim rel|⩽ \leqslant rel|⩾ \geqslant rel
± \pm bin|∓ \mp bin|× \times bin|÷ \div bin|· \cdot bin|∘ \circ bin|◦ \circ bin|∗ \ast bin|⋆ \star bin|• \bullet bin
∪ \cup bin|∩ \cap bin|⊔ \sqcup bin|⊓ \sqcap bin|⊎ \uplus bin|∧ \wedge bin|∨ \vee bin|∖ \setminus bin|⊕ \oplus bin|⊖ \ominus bin
⊗ \otimes bin|⊘ \oslash bin|⊙ \odot bin|† \dagger bin|‡ \ddagger bin|⋄ \diamond bin|△ \bigtriangleup bin|▽ \bigtriangledown bin
⨿ \amalg bin|≀ \wr bin|⊠ \boxtimes bin|⊞ \boxplus bin
→ \to arrow|← \leftarrow arrow|↔ \leftrightarrow arrow|↑ \uparrow arrow|↓ \downarrow arrow|↕ \updownarrow arrow
⇒ \Rightarrow arrow|⇐ \Leftarrow arrow|⇔ \Leftrightarrow arrow|⇑ \Uparrow arrow|⇓ \Downarrow arrow|↦ \mapsto arrow
⟶ \longrightarrow arrow|⟵ \longleftarrow arrow|⟷ \longleftrightarrow arrow|⟹ \Longrightarrow arrow|⟸ \Longleftarrow arrow
⟺ \Longleftrightarrow arrow|⟼ \longmapsto arrow|↪ \hookrightarrow arrow|↩ \hookleftarrow arrow|↠ \twoheadrightarrow arrow
↞ \twoheadleftarrow arrow|↣ \rightarrowtail arrow|↷ \curvearrowright arrow|↶ \curvearrowleft arrow|↗ \nearrow arrow
↘ \searrow arrow|↖ \nwarrow arrow|↙ \swarrow arrow|⇀ \rightharpoonup arrow|⇁ \rightharpoondown arrow|↝ \leadsto arrow
⇝ \rightsquigarrow arrow|⇄ \rightleftarrows arrow|⇉ \rightrightarrows arrow|⇌ \rightleftharpoons arrow|⇢ \dashrightarrow arrow
∀ \forall log|∃ \exists log|∄ \nexists log|¬ \neg log|∴ \therefore log|∵ \because log|⊤ \top log|⊥ \bot log
∆ \Delta greek|µ \mu greek|Ω \Omega greek|∞ \infty ord|∅ \emptyset ord|∂ \partial ord|∇ \nabla ord|ℓ \ell ord|ℏ \hbar ord|ℵ \aleph ord|ℜ \Re ord|ℑ \Im ord|℘ \wp ord
′ \prime ord|″ \prime\prime ord|‴ \prime\prime\prime ord|√ \surd ord|∠ \angle ord|♭ \flat ord|♮ \natural ord|♯ \sharp ord
∑ \sum op|∏ \prod op|∐ \coprod op|∫ \int op|∬ \iint op|∮ \oint op|⋂ \bigcap op|⋃ \bigcup op|⨆ \bigsqcup op|⨁ \bigoplus op|⨂ \bigotimes op
⨄ \biguplus op|⋁ \bigvee op|⋀ \bigwedge op
⟨ \langle open|⟩ \rangle close|⌊ \lfloor open|⌋ \rfloor close|⌈ \lceil open|⌉ \rceil close|‖ \Vert ord
… \ldots punct|⋯ \cdots punct|⋮ \vdots punct|⋱ \ddots punct|ℝ \mathbb{R} letter|ℤ \mathbb{Z} letter|ℚ \mathbb{Q} letter
ℂ \mathbb{C} letter|ℕ \mathbb{N} letter|ℙ \mathbb{P} letter|ℍ \mathbb{H} letter|ℓ \ell ord|□ \square ord|■ \blacksquare ord
§ \S ord|¶ \P ord|♣ \clubsuit ord|♢ \diamondsuit ord|♡ \heartsuit ord|♠ \spadesuit ord|∟ \llcorner ord|⊾ \measuredangle ord
"""
UNICODE_MACROS: dict[str, tuple[str, str]] = {}
for chunk in _RAW.replace("\n", "|").split("|"):
    chunk = chunk.strip()
    if chunk:
        ch, macro, cls = chunk.split(" ")
        UNICODE_MACROS.setdefault(ch, (macro, cls))
UNICODE_MACROS["−"] = ("-", "bin")           # U+2212 minus
UNICODE_MACROS["∣"] = ("\\mid", "rel")

# ASCII punctuation / delimiters that need escaping or are classed specially in math
ASCII_MATH = {
    "{": ("\\{", "open"), "}": ("\\}", "close"), "\\": ("\\backslash", "bin"), "%": ("\\%", "ord"), "&": ("\\&", "ord"),
    "#": ("\\#", "ord"), "_": ("\\_", "ord"), "$": ("\\$", "ord"), "~": ("\\sim", "rel"), "^": ("\\hat{}", "ord"),
    "(": ("(", "open"), ")": (")", "close"), "[": ("[", "open"), "]": ("]", "close"),
    "=": ("=", "rel"), "<": ("<", "rel"), ">": (">", "rel"), "+": ("+", "bin"), "-": ("-", "bin"), "|": ("|", "ord"),
    "/": ("/", "bin"), "*": ("*", "bin"), ",": (",", "punct"), ";": (";", "punct"), ":": (":", "rel"), ".": (".", "punct"),
    "!": ("!", "ord"), "?": ("?", "ord"), "'": ("'", "ord"), "@": ("@", "ord"),
}

# negation overlay (U+0338, or TeX \not) + relation -> macro
NEGATED = {
    "=": "\\neq", "∈": "\\notin", "∋": "\\not\\ni", "⊂": "\\not\\subset", "⊃": "\\not\\supset", "⊆": "\\nsubseteq",
    "⊇": "\\nsupseteq", "∼": "\\nsim", "≅": "\\ncong", "≡": "\\not\\equiv", "<": "\\nless", ">": "\\ngtr",
    "≤": "\\nleq", "≥": "\\ngeq", "→": "\\not\\to", "∃": "\\nexists", "∣": "\\nmid", "|": "\\nmid", "≈": "\\not\\approx",
    "⊴": "\\ntrianglelefteq", "⊵": "\\ntrianglerighteq", "∥": "\\nparallel", "≃": "\\not\\simeq", "◁": "\\not\\lhd",
}

ACCENTS = {"¯": "\\bar", "ˆ": "\\hat", "^": "\\hat", "˜": "\\tilde", "~": "\\tilde", "˙": "\\dot", "¨": "\\ddot",
           "´": "\\acute", "`": "\\grave", "˘": "\\breve", "ˇ": "\\check", "\u20d7": "\\vec"}

# roman words that are standard LaTeX operators
OPERATOR_MACROS = {w: "\\" + w for w in (
    "arccos arcsin arctan arg cos cosh cot coth csc deg det dim exp gcd hom inf ker lg lim liminf limsup ln log max min "
    "Pr sec sin sinh sup tan tanh").split()}

# ---------- font families -------------------------------------------------------------------------
MATH_ITALIC = {"CMMI", "CMMIB", "LMMathItalic", "LMMathBoldItalic", "MathItalic", "MTMI", "MTMIB"}
MATH_SYMBOL = {"CMSY", "CMBSY", "CMEX", "LASY", "MSAM", "MSBM", "LMMathSymbols", "LMMathExtension", "MTSY", "MTEX",
               "MTSYN", "STIXMathJax", "Cambria", "EUFM", "EUFB", "EUSM", "EUSB", "RSFS", "BBOLD", "BBM", "STIX"}
ROMAN_TEXT = {"CMR", "CMBX", "CMTI", "CMSL", "CMSS", "CMSSBX", "CMTT", "CMCSC", "CMU", "LMRoman", "LMRomanDemi",
              "LMRomanSlanted", "LMSans", "LMMono"}
BLACKBOARD = {"MSBM", "BBOLD", "BBM"}
CALLIGRAPHIC = {"CMSY", "CMBSY", "RSFS", "LMMathSymbols", "MTSY"}
FRAKTUR = {"EUFM", "EUFB"}


def font_family(name: str) -> str:
    base = name.split("+")[-1]
    m = re.match(r"[A-Za-z]+", base)
    fam = m.group(0) if m else base
    for known in sorted(MATH_ITALIC | MATH_SYMBOL | ROMAN_TEXT, key=len, reverse=True):
        if fam.upper().startswith(known.upper()):
            return known
    return fam


def font_kind(fam: str) -> str:
    if fam in MATH_ITALIC:
        return "mathital"
    if fam in MATH_SYMBOL:
        return "mathsym"
    if fam in ROMAN_TEXT:
        return "roman"
    return "other"


def unicode_is_mathy(c: str) -> bool:
    """For non-TeX fonts (Word, Cambria Math, ...): classify by code point alone."""
    o = ord(c)
    return (0x2190 <= o <= 0x22FF or 0x2A00 <= o <= 0x2AFF or 0x27C0 <= o <= 0x27EF or 0x2980 <= o <= 0x29FF
            or 0x1D400 <= o <= 0x1D7FF or 0x0391 <= o <= 0x03C9 or 0x03D0 <= o <= 0x03F5 or c in UNICODE_MACROS
            and UNICODE_MACROS[c][1] not in ("punct",))


def decode_math_alnum(c: str) -> tuple[str, str] | None:
    """U+1D400.. mathematical alphanumerics -> (macro, class)."""
    o = ord(c)
    if not 0x1D400 <= o <= 0x1D7FF:
        return None
    import unicodedata
    name = unicodedata.name(c, "")
    m = re.match(r"MATHEMATICAL (BOLD ITALIC|BOLD SCRIPT|BOLD FRAKTUR|BOLD|ITALIC|SCRIPT|FRAKTUR|DOUBLE-STRUCK|SANS-SERIF|MONOSPACE) "
                 r"(CAPITAL|SMALL|DIGIT) (\w+)", name)
    if not m:
        return None
    style, kind, rest = m.groups()
    letter = rest if kind == "DIGIT" else (rest.upper() if kind == "CAPITAL" else rest.lower())
    if kind == "DIGIT":
        letter = str("ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split().index(rest))
    wrap = {"BOLD": "\\mathbf", "BOLD ITALIC": "\\boldsymbol", "ITALIC": None, "SCRIPT": "\\mathcal", "BOLD SCRIPT": "\\mathcal",
            "FRAKTUR": "\\mathfrak", "BOLD FRAKTUR": "\\mathfrak", "DOUBLE-STRUCK": "\\mathbb", "SANS-SERIF": "\\mathsf",
            "MONOSPACE": "\\mathtt"}[style]
    return ((f"{wrap}{{{letter}}}" if wrap else letter), "letter")

UNICODE_MACROS["\u2126"] = ("\\Omega", "greek")
UNICODE_MACROS["\u03a9"] = ("\\Omega", "greek")