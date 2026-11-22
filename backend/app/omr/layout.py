"""
Authoritative OMR Sheet Layout Configuration.
Single source of truth for both the PDF generator and the CV detector.
Guarantees geometric synchronization between printed sheets and detection coordinates.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

# Canonical Rectified Page Dimensions (A4 at 200 DPI: 1654 x 2338)
CANONICAL_WIDTH = 1654
CANONICAL_HEIGHT = 2338

# Standard A4 Points (72 DPI)
A4_POINTS_WIDTH = 595.28
A4_POINTS_HEIGHT = 841.89

# Scale factors: Canonical Pixels per ReportLab Point
SCALE_X = CANONICAL_WIDTH / A4_POINTS_WIDTH    # ~2.77853
SCALE_Y = CANONICAL_HEIGHT / A4_POINTS_HEIGHT  # ~2.77708


@dataclass
class BubbleLocation:
    question: int
    option: str  # 'A', 'B', 'C', 'D'
    cx: int      # Canonical pixel X
    cy: int      # Canonical pixel Y
    radius: int  # Canonical pixel radius
    # Point coordinates for ReportLab drawing
    cx_pt: float = 0.0
    cy_pt: float = 0.0
    radius_pt: float = 0.0


@dataclass
class RollDigitBubble:
    column_index: int  # 0 to 5
    digit: int         # 0 to 9
    cx: int
    cy: int
    radius: int
    cx_pt: float = 0.0
    cy_pt: float = 0.0
    radius_pt: float = 0.0


class OMRLayoutConfig:
    """
    Authoritative template layout configuration.
    Shared by both ReportLab PDF generators and OpenCV OMR detectors.
    """

    def __init__(
        self,
        question_count: int = 80,
        options_per_question: int = 4,
        template_type: str = "navodaya_80",
    ):
        self.width = CANONICAL_WIDTH
        self.height = CANONICAL_HEIGHT
        self.question_count = question_count
        self.options_per_question = options_per_question
        self.template_type = template_type

        # Marker geometry (centers in canonical pixels and points)
        self.markers: Dict[str, Tuple[int, int]] = {}
        self.markers_pt: Dict[str, Tuple[float, float, float, float]] = {}  # x, y_from_top, w, h
        self.marker_size = 78  # ~28 pt * 2.78 = ~78 px
        self.marker_size_pt = 28.0

        # QR Box
        self.qr_box: Dict[str, int] = {}
        self.qr_box_pt: Dict[str, float] = {}

        # Roll Number Bubbles and handwritten digit boxes
        self.roll_digit_count = 0
        self.roll_bubbles: List[RollDigitBubble] = []
        self.roll_boxes: List[Tuple[int, int, int, int]] = []

        # Main Test Booklet Number Bubbles and handwritten digit boxes
        self.booklet_digit_count = 0
        self.booklet_bubbles: List[RollDigitBubble] = []
        self.booklet_boxes: List[Tuple[int, int, int, int]] = []

        # Invigilator section bubbles: State (2), Taluk (2), Centre (2)
        self.state_bubbles: List[RollDigitBubble] = []
        self.taluk_bubbles: List[RollDigitBubble] = []
        self.centre_bubbles: List[RollDigitBubble] = []

        # Main Booklet Code (A, B, C, D) and Medium (English, Kannada)
        self.booklet_code_bubbles: Dict[str, Tuple[int, int]] = {}
        self.medium_bubbles: Dict[str, Tuple[int, int]] = {}

        # Candidate Name & Father Name boxes
        self.candidate_name_box: Dict[str, int] = {}
        self.father_name_box: Dict[str, int] = {}

        # Question Bubbles: {question_num: {option: BubbleLocation}}
        self.bubble_radius = 15  # canonical pixels
        self.bubble_radius_pt = 15.0 / SCALE_X
        self.bubbles: Dict[int, Dict[str, BubbleLocation]] = {}

        # Build layout geometry based on template type
        if self.template_type in ("navodaya_80", "navodaya") or self.question_count == 80:
            self.template_type = "navodaya_80"
            self._build_navodaya_80_layout()
        elif self.template_type in ("vidyakashi_official", "vidyakashi_100"):
            self._build_vidyakashi_official_layout()
        elif self.template_type in ("vidyakashi_25", "vidyakashi_50"):
            self._build_vidyakashi_layout()
        else:
            self._build_standard_layout()

    # ──────────────────────────────────────────────────────────────────────────
    # Official Navodaya 4-Column 80Q OMR Sheet Layout (Exact Calibrated Grid)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_navodaya_80_layout(self):
        """
        Constructs the authoritative calibrated geometry for the official 80-question
        Navodaya OMR Answer Sheet in canonical 1654 x 2338 space.
        Includes:
        - 4 Corner registration markers
        - 80 Questions in 4 columns (1-20, 21-40, 41-60, 61-80) with options A, B, C, D
        - 7-digit Roll Number bubbled matrix (0-9) + 7 handwritten digit boxes
        - 7-digit Main Test Booklet Number bubbled matrix (0-9) + 7 handwritten digit boxes
        - Candidate Name & Father Name boxes
        - State (2 cols), Taluk (2 cols), Centre (2 cols) invigilator code grids
        - Main Booklet Code (A, B, C, D) bubbles
        - Medium (English, Kannada) bubbles
        """
        self.question_count = 80
        self.options_per_question = 4
        self.bubble_radius = 15
        self.bubble_radius_pt = self.bubble_radius / SCALE_X
        self.marker_size = 36
        self.marker_size_pt = self.marker_size / SCALE_X

        # 1. Four corner black square registration markers (centers in canonical space)
        self.markers = {
            "top_left": (57, 57),
            "top_right": (1597, 57),
            "bottom_left": (57, 2281),
            "bottom_right": (1597, 2281),
        }
        for name, (cx, cy) in self.markers.items():
            w_pt = self.marker_size_pt
            h_pt = self.marker_size_pt
            self.markers_pt[name] = (
                (cx / SCALE_X) - (w_pt / 2.0),
                (cy / SCALE_Y) - (h_pt / 2.0),
                w_pt,
                h_pt,
            )

        # 2. Roll Number: 7 columns of bubbled digits (0 to 9) and 7 handwritten digit boxes
        self.roll_digit_count = 7
        self.roll_bubbles = []
        self.roll_boxes = []
        roll_col_xs = [138, 179, 219, 260, 301, 342, 383]
        digit_row_ys = [438, 470, 502, 534, 566, 598, 630, 662, 694, 726]

        for c_idx, cx in enumerate(roll_col_xs):
            bx1 = int(round(116 + c_idx * 41.7))
            bx2 = int(round(116 + (c_idx + 1) * 41.7))
            self.roll_boxes.append((bx1, 340, bx2, 405))
            for digit, cy in enumerate(digit_row_ys):
                self.roll_bubbles.append(
                    RollDigitBubble(
                        column_index=c_idx,
                        digit=digit,
                        cx=cx,
                        cy=cy,
                        radius=14,
                        cx_pt=cx / SCALE_X,
                        cy_pt=cy / SCALE_Y,
                        radius_pt=14.0 / SCALE_X,
                    )
                )

        # 3. Main Test Booklet No: 7 columns of bubbled digits (0 to 9) and 7 handwritten digit boxes
        self.booklet_digit_count = 7
        self.booklet_bubbles = []
        self.booklet_boxes = []
        booklet_col_xs = [450, 491, 532, 573, 614, 655, 696]

        for c_idx, cx in enumerate(booklet_col_xs):
            bx1 = int(round(428 + c_idx * 41.7))
            bx2 = int(round(428 + (c_idx + 1) * 41.7))
            self.booklet_boxes.append((bx1, 340, bx2, 405))
            for digit, cy in enumerate(digit_row_ys):
                self.booklet_bubbles.append(
                    RollDigitBubble(
                        column_index=c_idx,
                        digit=digit,
                        cx=cx,
                        cy=cy,
                        radius=14,
                        cx_pt=cx / SCALE_X,
                        cy_pt=cy / SCALE_Y,
                        radius_pt=14.0 / SCALE_X,
                    )
                )

        # 4. Candidate Name & Father's Name boxes
        self.candidate_name_box = {"x1": 775, "y1": 315, "x2": 1130, "y2": 515}
        self.father_name_box = {"x1": 775, "y1": 575, "x2": 1130, "y2": 778}

        # 5. Invigilator Section: State (2 cols), Taluk (2 cols), Centre (2 cols)
        self.state_bubbles = []
        for c_idx, cx in enumerate([1218, 1259]):
            for digit, cy in enumerate(digit_row_ys):
                self.state_bubbles.append(
                    RollDigitBubble(column_index=c_idx, digit=digit, cx=cx, cy=cy, radius=14)
                )

        self.taluk_bubbles = []
        for c_idx, cx in enumerate([1340, 1381]):
            for digit, cy in enumerate(digit_row_ys):
                self.taluk_bubbles.append(
                    RollDigitBubble(column_index=c_idx, digit=digit, cx=cx, cy=cy, radius=14)
                )

        self.centre_bubbles = []
        for c_idx, cx in enumerate([1466, 1507]):
            for digit, cy in enumerate(digit_row_ys):
                self.centre_bubbles.append(
                    RollDigitBubble(column_index=c_idx, digit=digit, cx=cx, cy=cy, radius=14)
                )

        # 6. Main Booklet Code (A, B, C, D)
        self.booklet_code_bubbles = {
            "A": (282, 1472),
            "B": (282, 1526),
            "C": (282, 1578),
            "D": (282, 1632),
        }

        # 7. Medium (English, Kannada)
        self.medium_bubbles = {
            "English": (304, 1788),
            "Kannada": (304, 1834),
        }

        # 8. All 80 Questions in 4 columns:
        # Col 0: Q1..20
        # Col 1: Q21..40
        # Col 2: Q41..60
        # Col 3: Q61..80
        block_start_xs = [533.5, 820.0, 1106.5, 1393.0]
        opt_step = 40.0
        row_ys = [
            952.5, 997.5, 1043.5, 1088.5, 1133.5, 1181.5, 1227.5, 1272.5, 1318.5, 1363.5,
            1410.5, 1455.5, 1501.0, 1546.5, 1591.5, 1637.0, 1682.5, 1728.0, 1773.5, 1818.5,
        ]
        options = ["A", "B", "C", "D"]

        self.bubbles = {}
        for b_idx in range(4):
            for r_idx in range(20):
                q_num = b_idx * 20 + r_idx + 1
                cy_px = int(round(row_ys[r_idx]))
                cy_pt = cy_px / SCALE_Y

                self.bubbles[q_num] = {}
                for opt_idx, opt in enumerate(options):
                    cx_px = int(round(block_start_xs[b_idx] + opt_idx * opt_step))
                    cx_pt = cx_px / SCALE_X

                    self.bubbles[q_num][opt] = BubbleLocation(
                        question=q_num,
                        option=opt,
                        cx=cx_px,
                        cy=cy_px,
                        radius=self.bubble_radius,
                        cx_pt=cx_pt,
                        cy_pt=cy_pt,
                        radius_pt=self.bubble_radius_pt,
                    )

    # ──────────────────────────────────────────────────────────────────────────
    # Official Vidyakashi 4-Column 100Q OMR Sheet Layout (Exact Calibrated Grid)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_vidyakashi_official_layout(self):
        """
        Constructs the exact calibrated geometry for the official Vidyakashi Study Circle
        4-column (100-question) OMR Answer Sheet in canonical 1654 x 2338 space.
        Supports 25, 50, 75, or 100 questions mapped to the 4 vertical blocks (25 rows each).
        """
        self.bubble_radius = 15
        self.bubble_radius_pt = self.bubble_radius / SCALE_X
        self.marker_size = 36
        self.marker_size_pt = self.marker_size / SCALE_X

        # Outer corners of the 4-column question grid box for visual reference
        self.markers = {
            "top_left": (28, 938),
            "top_right": (1626, 938),
            "bottom_left": (28, 2068),
            "bottom_right": (1626, 2068),
        }
        for name, (cx, cy) in self.markers.items():
            w_pt = self.marker_size_pt
            h_pt = self.marker_size_pt
            self.markers_pt[name] = (
                (cx / SCALE_X) - (w_pt / 2.0),
                (cy / SCALE_Y) - (h_pt / 2.0),
                w_pt,
                h_pt,
            )

        self.qr_box = {"x": 1420, "y": 2120, "width": 160, "height": 160}
        self.qr_box_pt = {
            "x": 1420 / SCALE_X,
            "y": 2120 / SCALE_Y,
            "width": 160 / SCALE_X,
            "height": 160 / SCALE_Y,
        }

        # Vidyakashi uses 9 handwritten Roll No. boxes (no roll bubble grid)
        self.roll_digit_count = 0
        self.roll_bubbles = []

        # Authoritative calibrated X centers for each of the 4 blocks (Q1-25, Q26-50, Q51-75, Q76-100)
        block_option_x = [
            {"A": 204, "B": 267, "C": 328, "D": 389},      # Block 0: Q01 - Q25
            {"A": 608, "B": 668, "C": 730, "D": 790},      # Block 1: Q26 - Q50
            {"A": 1000, "B": 1061, "C": 1123, "D": 1185},  # Block 2: Q51 - Q75
            {"A": 1404, "B": 1466, "C": 1527, "D": 1589},  # Block 3: Q76 - Q100
        ]

        # Authoritative calibrated Y centers for the 25 rows in each block
        row_y = [
            1023, 1067, 1109, 1152, 1195,
            1238, 1281, 1324, 1367, 1410,
            1452, 1495, 1538, 1580, 1623,
            1665, 1708, 1750, 1792, 1835,
            1877, 1920, 1962, 2004, 2047,
        ]

        options = ["A", "B", "C", "D"][: self.options_per_question]
        total_q = min(self.question_count, 100)

        for q in range(1, total_q + 1):
            block_idx = (q - 1) // 25
            row_idx = (q - 1) % 25
            cy_px = row_y[row_idx]
            cy_pt = cy_px / SCALE_Y

            self.bubbles[q] = {}
            for opt in options:
                cx_px = block_option_x[block_idx][opt]
                cx_pt = cx_px / SCALE_X
                self.bubbles[q][opt] = BubbleLocation(
                    question=q,
                    option=opt,
                    cx=cx_px,
                    cy=cy_px,
                    radius=self.bubble_radius,
                    cx_pt=cx_pt,
                    cy_pt=cy_pt,
                    radius_pt=self.bubble_radius_pt,
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Vidyakashi Branded 2-Column Layout
    # ──────────────────────────────────────────────────────────────────────────

    def _build_vidyakashi_layout(self):
        """Constructs the exact geometry for Vidyakashi Study Circle sheets."""
        # 1. Four Corner Registration Markers
        # 18 pt margin from page edges, size 28x28 pt
        # Leaving > 20 pt clearance from the bottom office-use box
        margin_pt = 18.0
        size_pt = self.marker_size_pt  # 28.0 pt

        # Top-Left: x=18, y_from_top=18
        # Top-Right: x=595.28 - 18 - 28 = 549.28, y_from_top=18
        # Bottom-Left: x=18, y_from_top=841.89 - 18 - 28 = 795.89
        # Bottom-Right: x=549.28, y_from_top=795.89
        self.markers_pt = {
            "top_left": (margin_pt, margin_pt, size_pt, size_pt),
            "top_right": (A4_POINTS_WIDTH - margin_pt - size_pt, margin_pt, size_pt, size_pt),
            "bottom_left": (margin_pt, A4_POINTS_HEIGHT - margin_pt - size_pt, size_pt, size_pt),
            "bottom_right": (A4_POINTS_WIDTH - margin_pt - size_pt, A4_POINTS_HEIGHT - margin_pt - size_pt, size_pt, size_pt),
        }

        # Marker centers in canonical pixels
        for name, (x_pt, y_pt, w_pt, h_pt) in self.markers_pt.items():
            cx = int(round((x_pt + w_pt / 2.0) * SCALE_X))
            cy = int(round((y_pt + h_pt / 2.0) * SCALE_Y))
            self.markers[name] = (cx, cy)

        # 2. QR Code (Bottom Right in Invigilator area)
        self.qr_box_pt = {
            "x": A4_POINTS_WIDTH - margin_pt - size_pt - 46.0,
            "y": A4_POINTS_HEIGHT - margin_pt - size_pt - 2.0,
            "width": 42.0,
            "height": 42.0,
        }
        self.qr_box = {
            "x": int(round(self.qr_box_pt["x"] * SCALE_X)),
            "y": int(round(self.qr_box_pt["y"] * SCALE_Y)),
            "width": int(round(self.qr_box_pt["width"] * SCALE_X)),
            "height": int(round(self.qr_box_pt["height"] * SCALE_Y)),
        }

        # Roll Number: Vidyakashi uses handwritten digit boxes, no bubble grid
        self.roll_digit_count = 0
        self.roll_bubbles = []

        # 3. Two-Column Question Bubble Grid
        banner_x = 24.0
        banner_w = A4_POINTS_WIDTH - 48.0
        grid_start_y = 332.0
        grid_w = (banner_w - 12.0) / 2.0  # 267.64 pt
        table_left_x = banner_x           # 24.0 pt
        table_right_x = banner_x + grid_w + 12.0  # 303.64 pt

        row_h = 28.5
        col_w_qno = 54.0
        col_w_opt = (grid_w - col_w_qno) / 4.0  # 53.41 pt
        options = ["A", "B", "C", "D"][: self.options_per_question]

        total_q = min(self.question_count, 50)
        split_q = 13 if total_q <= 25 else 25

        for q in range(1, total_q + 1):
            if q <= split_q:
                # Column 1 (Left Table)
                base_x = table_left_x
                r = q
            else:
                # Column 2 (Right Table)
                base_x = table_right_x
                r = q - split_q

            y_center_pt = grid_start_y + (r * row_h) + (row_h / 2.0)
            cy_px = int(round(y_center_pt * SCALE_Y))

            self.bubbles[q] = {}
            for opt_idx, opt in enumerate(options):
                x_center_pt = base_x + col_w_qno + (opt_idx * col_w_opt) + (col_w_opt / 2.0)
                cx_px = int(round(x_center_pt * SCALE_X))

                self.bubbles[q][opt] = BubbleLocation(
                    question=q,
                    option=opt,
                    cx=cx_px,
                    cy=cy_px,
                    radius=self.bubble_radius,
                    cx_pt=x_center_pt,
                    cy_pt=y_center_pt,
                    radius_pt=self.bubble_radius_pt,
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Standard Multi-Column Layout (for 40, 50, 100, 150, 200 questions)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_standard_layout(self):
        """Constructs the standard multi-column layout with 6-digit roll number bubble grid."""
        margin_px = 70
        self.marker_size = 48
        self.marker_size_pt = self.marker_size / SCALE_X

        self.markers = {
            "top_left": (margin_px, margin_px),
            "top_right": (self.width - margin_px, margin_px),
            "bottom_left": (margin_px, self.height - margin_px),
            "bottom_right": (self.width - margin_px, self.height - margin_px),
        }

        # QR Box (Top Right)
        self.qr_box = {
            "x": self.width - 290,
            "y": 140,
            "width": 180,
            "height": 180,
        }

        # 6-digit Roll Number Bubble Matrix
        self.roll_digit_count = 6
        start_x = 760
        start_y = 175
        col_spacing = 38
        row_spacing = 22
        radius = 9

        for col in range(self.roll_digit_count):
            cx = start_x + (col * col_spacing)
            for digit in range(10):
                cy = start_y + (digit * row_spacing)
                self.roll_bubbles.append(
                    RollDigitBubble(
                        column_index=col,
                        digit=digit,
                        cx=cx,
                        cy=cy,
                        radius=radius,
                        cx_pt=cx / SCALE_X,
                        cy_pt=cy / SCALE_Y,
                        radius_pt=radius / SCALE_X,
                    )
                )

        # Question grid
        options = ["A", "B", "C", "D"][: self.options_per_question]
        if self.question_count <= 25:
            num_cols = 1
            rows_per_col = self.question_count
        elif self.question_count <= 50:
            num_cols = 2
            rows_per_col = (self.question_count + 1) // 2
        elif self.question_count <= 100:
            num_cols = 4
            rows_per_col = (self.question_count + 3) // 4
        else:
            num_cols = 5
            rows_per_col = (self.question_count + 4) // 5

        grid_start_y = 440
        grid_start_x = 100
        available_width = self.width - 200
        col_width = available_width // num_cols
        available_height = self.height - grid_start_y - 120
        row_height = min(42, available_height // rows_per_col)
        bubble_spacing = 28
        self.bubble_radius = 14
        self.bubble_radius_pt = self.bubble_radius / SCALE_X

        q_num = 1
        for col in range(num_cols):
            base_col_x = grid_start_x + (col * col_width)
            for row in range(rows_per_col):
                if q_num > self.question_count:
                    break
                q_y = grid_start_y + (row * row_height) + 18
                self.bubbles[q_num] = {}
                bubble_start_x = base_col_x + 65
                for opt_idx, opt in enumerate(options):
                    bx = bubble_start_x + (opt_idx * bubble_spacing)
                    self.bubbles[q_num][opt] = BubbleLocation(
                        question=q_num,
                        option=opt,
                        cx=bx,
                        cy=q_y,
                        radius=self.bubble_radius,
                        cx_pt=bx / SCALE_X,
                        cy_pt=q_y / SCALE_Y,
                        radius_pt=self.bubble_radius_pt,
                    )
                q_num += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serializes layout configuration to JSON-compatible dictionary."""
        return {
            "template_type": self.template_type,
            "canonical_width": self.width,
            "canonical_height": self.height,
            "question_count": self.question_count,
            "options_per_question": self.options_per_question,
            "markers": self.markers,
            "marker_size": self.marker_size,
            "qr_box": self.qr_box,
            "bubble_radius": self.bubble_radius,
            "questions": {
                str(q): {
                    opt: {"cx": b.cx, "cy": b.cy, "radius": b.radius}
                    for opt, b in opts.items()
                }
                for q, opts in self.bubbles.items()
            },
            "roll_bubbles": [
                {
                    "col": r.column_index,
                    "digit": r.digit,
                    "cx": r.cx,
                    "cy": r.cy,
                    "radius": r.radius,
                }
                for r in self.roll_bubbles
            ],
            "booklet_bubbles": [
                {
                    "col": r.column_index,
                    "digit": r.digit,
                    "cx": r.cx,
                    "cy": r.cy,
                    "radius": r.radius,
                }
                for r in self.booklet_bubbles
            ],
            "booklet_code_bubbles": self.booklet_code_bubbles,
            "medium_bubbles": self.medium_bubbles,
            "candidate_name_box": self.candidate_name_box,
            "father_name_box": self.father_name_box,
        }
