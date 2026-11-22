"""
Robust Computer Vision OMR Detection Engine with Sensitive Small-Mark Recognition.
Implements the complete OpenCV image processing pipeline with:
- Dual-mode corner registration marker and document boundary detection
- Perspective correction and canonical alignment
- Inner-bubble ink measurement strictly excluding printed circular borders
- Multi-feature small-mark detection (Connected Component Analysis, Dark Pixel Ratio, Local Contrast)
- Support for small dots, partial pen strokes, faint pencil marks, partial fills, and solid fills
- Invariance to ambient room lighting, flash reflections, shadows, and paper grain
- Administrator-configurable sensitivity presets (HIGH, NORMAL, CONSERVATIVE)
- Calibrated response classification (Single, Blank, Multiple, Review)
- Visual diagnostic overlay generation with mark centroid localization for human inspection
"""
import cv2
import math
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from backend.app.config import settings
from backend.app.omr.preprocessor import ImagePreprocessor
from backend.app.omr.htr_engine import HTREngine
from backend.app.omr.layout import (
    OMRLayoutConfig,
    BubbleLocation,
    CANONICAL_WIDTH,
    CANONICAL_HEIGHT,
    SCALE_X,
    SCALE_Y,
)


class OMRDetector:
    def __init__(
        self,
        layout: Optional[OMRLayoutConfig] = None,
        sensitivity: Optional[str] = None,
    ):
        self.layout = layout or OMRLayoutConfig(question_count=80, template_type="navodaya_80")
        self.expected_question_count = self.layout.question_count
        self._last_alignment_info: Dict[str, Any] = {}
        self._last_oriented_img: Optional[np.ndarray] = None
        self.sensitivity = (sensitivity or getattr(settings, "OMR_SENSITIVITY", "NORMAL")).upper()
        if self.sensitivity not in ("HIGH", "NORMAL", "CONSERVATIVE", "LOW"):
            self.sensitivity = "NORMAL"

        # Calibrated legacy thresholds for backward compatibility
        self.blank_threshold = 0.20
        self.fill_threshold = 0.38
        self.multiple_threshold = 0.30
        self.contrast_margin = 0.18

    def set_sensitivity(self, sensitivity: str) -> None:
        """Sets the detection sensitivity level ('HIGH', 'NORMAL', 'CONSERVATIVE')."""
        sens = sensitivity.upper()
        if sens in ("HIGH", "NORMAL", "CONSERVATIVE", "LOW"):
            self.sensitivity = "CONSERVATIVE" if sens == "LOW" else sens

    def order_quad_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders 4 points as: top-left, top-right, bottom-right, bottom-left."""
        pts = np.asarray(pts, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum (x + y)
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum (x + y)

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff (y - x)
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff (y - x)
        return rect

    def find_registration_markers(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects the 4 corner registration markers on the sheet.
        Supports both square and rectangular blocks across varying lighting conditions.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        page_area = float(h * w)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (page_area * 0.00008) or area > (page_area * 0.035):
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw == 0 or ch == 0:
                continue

            aspect = float(cw) / float(ch)
            solidity = area / float(cw * ch)

            if 0.35 <= aspect <= 2.8 and solidity >= 0.58:
                roi_gray = gray[y : y + ch, x : x + cw]
                if roi_gray.size == 0 or float(np.mean(roi_gray)) > 125.0:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    # Corner markers reside near the 4 outer corners (allowing for tilt/rotation)
                    if (cx < w * 0.24 or cx > w * 0.76) and (cy < h * 0.18 or cy > h * 0.82):
                        candidates.append((cx, cy, area))

        if len(candidates) < 4:
            return None

        mid_x, mid_y = w / 2.0, h / 2.0
        tl_cands = [c for c in candidates if c[0] < mid_x and c[1] < mid_y]
        tr_cands = [c for c in candidates if c[0] >= mid_x and c[1] < mid_y]
        br_cands = [c for c in candidates if c[0] >= mid_x and c[1] >= mid_y]
        bl_cands = [c for c in candidates if c[0] < mid_x and c[1] >= mid_y]

        if not (tl_cands and tr_cands and br_cands and bl_cands):
            return None

        tl = min(tl_cands, key=lambda p: (p[0]**2 + p[1]**2))
        tr = min(tr_cands, key=lambda p: ((w - p[0])**2 + p[1]**2))
        br = min(br_cands, key=lambda p: ((w - p[0])**2 + (h - p[1])**2))
        bl = min(bl_cands, key=lambda p: (p[0]**2 + (h - p[1])**2))

        areas = [tl[2], tr[2], br[2], bl[2]]
        if max(areas) / max(1.0, min(areas)) > 3.5:
            return None

        return np.array([(tl[0], tl[1]), (tr[0], tr[1]), (br[0], br[1]), (bl[0], bl[1])], dtype="float32")

    def detect_sheet_boundary(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects the outer boundary of the A4 page on contrasting backgrounds.
        Used as a robust fallback/reinforcement when corner markers are cropped or obscured.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = float(h * w)

        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < (img_area * 0.35):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype("float32")
                ordered = self.order_quad_points(pts)
                min_y = float(np.min(ordered[:, 1]))
                max_y = float(np.max(ordered[:, 1]))
                # Must span nearly the full document height (rejects internal form boxes)
                if min_y < (h * 0.08) and max_y > (h * 0.92):
                    return ordered

        return None

    def auto_calibrate_vidyakashi_grid(self, canon_img: np.ndarray) -> bool:
        """
        Automatically calibrates the exact X-columns, Y-rows, and bubble radius
        directly from the detected bubble circles on a canonical (1654 x 2338) Vidyakashi sheet.
        Supports both 4-column (100Q) and 2-column (25Q) Vidyakashi variants.
        """
        gray = cv2.cvtColor(canon_img, cv2.COLOR_BGR2GRAY) if len(canon_img.shape) == 3 else canon_img
        if len(canon_img.shape) == 3:
            b, g, r = cv2.split(canon_img)
            redness = np.clip(
                r.astype(np.float32) - 0.5 * (g.astype(np.float32) + b.astype(np.float32)),
                0,
                255,
            ).astype(np.uint8)
        else:
            redness = np.zeros_like(gray)

        y_min, y_max = 1015, 2080
        roi_gray = gray[y_min:y_max, :]
        roi_red = redness[y_min:y_max, :]

        _, th_dark = cv2.threshold(roi_gray, 140, 255, cv2.THRESH_BINARY_INV)
        _, th_red = cv2.threshold(roi_red, 20, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(th_dark, th_red)

        contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        raw_circles = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 20 <= cw <= 56 and 20 <= ch <= 56:
                aspect = cw / float(ch)
                if 0.80 <= aspect <= 1.25:
                    area = cv2.contourArea(cnt)
                    if area >= 120:
                        cx = x + cw / 2.0
                        cy = y + y_min + ch / 2.0
                        rad = (cw + ch) / 4.0
                        raw_circles.append((cx, cy, rad))

        dedup = []
        for cx, cy, rad in sorted(raw_circles, key=lambda c: c[2], reverse=True):
            if not any(abs(cx - dx) < 12 and abs(cy - dy) < 12 for dx, dy, _ in dedup):
                dedup.append((cx, cy, rad))

        if len(dedup) < 40:
            return False

        xs = sorted([c[0] for c in dedup])
        x_groups = []
        if xs:
            curr = [xs[0]]
            for x in xs[1:]:
                if x - curr[-1] > 20:
                    x_groups.append(curr)
                    curr = [x]
                else:
                    curr.append(x)
            x_groups.append(curr)
        valid_x_cols = [int(round(np.median(grp))) for grp in x_groups if len(grp) >= 6]

        ys = sorted([c[1] for c in dedup])
        y_groups = []
        if ys:
            curr = [ys[0]]
            for y in ys[1:]:
                if y - curr[-1] > 16:
                    y_groups.append(curr)
                    curr = [y]
                else:
                    curr.append(y)
            y_groups.append(curr)
        valid_y_rows = [int(round(np.median(grp))) for grp in y_groups if len(grp) >= 3]
        median_rad = int(round(np.median([c[2] for c in dedup]))) if dedup else 15

        options = ["A", "B", "C", "D"][: self.layout.options_per_question]

        if len(valid_x_cols) == 16 and len(valid_y_rows) >= 25:
            row_y = valid_y_rows[-25:]
            self.layout = OMRLayoutConfig(
                question_count=self.layout.question_count,
                options_per_question=self.layout.options_per_question,
                template_type="vidyakashi_official",
            )
            self.layout.bubble_radius = median_rad
            self.layout.bubbles = {}
            total_q = min(self.layout.question_count, 100)
            for q in range(1, total_q + 1):
                b_idx = (q - 1) // 25
                r_idx = (q - 1) % 25
                cy_px = row_y[r_idx]
                self.layout.bubbles[q] = {}
                for opt_i, opt in enumerate(options):
                    cx_px = valid_x_cols[b_idx * 4 + opt_i]
                    self.layout.bubbles[q][opt] = BubbleLocation(
                        question=q,
                        option=opt,
                        cx=cx_px,
                        cy=cy_px,
                        radius=median_rad,
                        cx_pt=cx_px / SCALE_X,
                        cy_pt=cy_px / SCALE_Y,
                        radius_pt=median_rad / SCALE_X,
                    )
            return True

        if len(valid_x_cols) == 8 and len(valid_y_rows) >= 13:
            row_y = valid_y_rows[-13:]
            self.layout = OMRLayoutConfig(
                question_count=self.layout.question_count,
                options_per_question=self.layout.options_per_question,
                template_type="vidyakashi_25",
            )
            self.layout.bubble_radius = median_rad
            self.layout.bubbles = {}
            total_q = min(self.layout.question_count, 25)
            for q in range(1, total_q + 1):
                b_idx = 0 if q <= 13 else 1
                r_idx = (q - 1) if q <= 13 else (q - 14)
                cy_px = row_y[r_idx]
                self.layout.bubbles[q] = {}
                for opt_i, opt in enumerate(options):
                    cx_px = valid_x_cols[b_idx * 4 + opt_i]
                    self.layout.bubbles[q][opt] = BubbleLocation(
                        question=q,
                        option=opt,
                        cx=cx_px,
                        cy=cy_px,
                        radius=median_rad,
                        cx_pt=cx_px / SCALE_X,
                        cy_pt=cy_px / SCALE_Y,
                        radius_pt=median_rad / SCALE_X,
                    )
            return True

        return False

    def orient_and_deskew_image(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Automatically corrects:
        - 90° / 270° landscape orientation
        - 180° upside-down orientation
        - Minor in-plane rotation / tilt (1° to 28°)
        Returns (oriented_img, alignment_metadata).
        """
        oriented = img.copy()
        h, w = oriented.shape[:2]
        rotation_deg = 0
        tilt_angle_deg = 0.0
        was_tilted = False

        # 1. Landscape orientation check (90° or 270° rotated sheet)
        if w > h * 1.12:
            rot_cw = cv2.rotate(oriented, cv2.ROTATE_90_CLOCKWISE)
            rot_ccw = cv2.rotate(oriented, cv2.ROTATE_90_COUNTERCLOCKWISE)

            def _header_advantage(cand: np.ndarray) -> float:
                g = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY) if len(cand.shape) == 3 else cand
                ch, cw = g.shape[:2]
                top_band = g[int(ch * 0.02):int(ch * 0.22), int(cw * 0.12):int(cw * 0.88)]
                bot_band = g[int(ch * 0.78):int(ch * 0.96), int(cw * 0.12):int(cw * 0.88)]
                top_m = float(np.mean(top_band)) if top_band.size > 0 else 255.0
                bot_m = float(np.mean(bot_band)) if bot_band.size > 0 else 255.0
                return bot_m - top_m

            if _header_advantage(rot_cw) >= _header_advantage(rot_ccw):
                oriented = rot_cw
                rotation_deg = 90
            else:
                oriented = rot_ccw
                rotation_deg = 270
            was_tilted = True

        # 2. Upside-down (180°) check on portrait image
        gray_p = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY) if len(oriented.shape) == 3 else oriented
        ph, pw = gray_p.shape[:2]
        top_banner = gray_p[int(ph * 0.02):int(ph * 0.14), int(pw * 0.15):int(pw * 0.85)]
        bot_banner = gray_p[int(ph * 0.86):int(ph * 0.98), int(pw * 0.15):int(pw * 0.85)]
        top_mean = float(np.mean(top_banner)) if top_banner.size > 0 else 255.0
        bot_mean = float(np.mean(bot_banner)) if bot_banner.size > 0 else 255.0

        # Also check upper vs lower header density for standard sheets
        top_header = gray_p[int(ph * 0.04):int(ph * 0.20), int(pw * 0.10):int(pw * 0.45)]
        bot_header = gray_p[int(ph * 0.80):int(ph * 0.96), int(pw * 0.55):int(pw * 0.90)]
        th_mean = float(np.mean(top_header)) if top_header.size > 0 else 255.0
        bh_mean = float(np.mean(bot_header)) if bot_header.size > 0 else 255.0

        if (bot_mean < 160.0 and top_mean > 185.0) or (th_mean > 250.5 and bh_mean < 242.0):
            oriented = cv2.rotate(oriented, cv2.ROTATE_180)
            rotation_deg = (rotation_deg + 180) % 360
            was_tilted = True
            gray_p = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY) if len(oriented.shape) == 3 else oriented
            ph, pw = gray_p.shape[:2]

        # 3. Estimate in-plane skew angle and deskew if tilted
        skew_deg = ImagePreprocessor.estimate_skew_angle(gray_p)
        if 1.0 <= abs(skew_deg) <= 28.0:
            tilt_angle_deg = round(float(skew_deg), 2)
            was_tilted = True
            banner_check = gray_p[int(ph * 0.02):int(ph * 0.14), int(pw * 0.15):int(pw * 0.85)]
            is_vidyakashi_banner = bool(banner_check.size > 0 and float(np.mean(banner_check)) < 155.0)
            # If Vidyakashi sheet or 4 corner markers aren't directly handling the warp, deskew via affine rotation
            if is_vidyakashi_banner or self.find_registration_markers(oriented) is None:
                border_val = (255, 255, 255) if len(oriented.shape) == 3 else 255

                def _rotate_by(src_img: np.ndarray, angle_deg: float) -> np.ndarray:
                    sh_i, sw_i = src_img.shape[:2]
                    mat = cv2.getRotationMatrix2D((sw_i / 2.0, sh_i / 2.0), angle_deg, 1.0)
                    return cv2.warpAffine(
                        src_img,
                        mat,
                        (sw_i, sh_i),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=border_val,
                    )

                cand_pos = _rotate_by(oriented, skew_deg)
                cand_neg = _rotate_by(oriented, -skew_deg)
                g_pos = cv2.cvtColor(cand_pos, cv2.COLOR_BGR2GRAY) if len(cand_pos.shape) == 3 else cand_pos
                g_neg = cv2.cvtColor(cand_neg, cv2.COLOR_BGR2GRAY) if len(cand_neg.shape) == 3 else cand_neg
                res_pos = abs(ImagePreprocessor.estimate_skew_angle(g_pos))
                res_neg = abs(ImagePreprocessor.estimate_skew_angle(g_neg))
                oriented = cand_pos if res_pos <= res_neg else cand_neg

        return oriented, {
            "was_tilted": was_tilted,
            "tilt_angle_deg": tilt_angle_deg,
            "rotation_deg": rotation_deg,
        }

    def rectify_image(self, img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Rectifies the input image to the canonical A4 coordinate space (1654 x 2338).
        Supports:
        - Automatic 90°/180°/270° orientation and in-plane tilt deskewing
        - Official Vidyakashi 4-Column 100Q OMR Sheet (vidyakashi_official) with auto-grid calibration
        - Vidyakashi 2-Column 25Q OMR Sheet (vidyakashi_25) with auto-grid calibration
        - Standard A4 OMR Sheet with 4 corner markers (standard)
        """
        oriented_img, align_meta = self.orient_and_deskew_image(img)
        self._last_oriented_img = oriented_img
        self._last_alignment_info = align_meta

        gray_sample = cv2.cvtColor(oriented_img, cv2.COLOR_BGR2GRAY) if len(oriented_img.shape) == 3 else oriented_img
        sh, sw = gray_sample.shape[:2]
        banner_crop = gray_sample[int(sh * 0.02):int(sh * 0.14), int(sw * 0.15):int(sw * 0.85)]
        has_dark_banner = bool(banner_crop.size > 0 and float(np.mean(banner_crop)) < 155.0)

        # 1. If Vidyakashi header banner is detected, try direct canonical resize + circle grid self-calibration first
        if has_dark_banner:
            resized = cv2.resize(oriented_img, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            if self.auto_calibrate_vidyakashi_grid(resized):
                return resized, True

        # 2. Try 4-corner black registration marker detection
        marker_pts = self.find_registration_markers(oriented_img)
        if marker_pts is not None:
            if has_dark_banner and self.layout.template_type not in ("vidyakashi_25", "vidyakashi_50") and self.layout.question_count <= 50:
                self.layout = OMRLayoutConfig(
                    question_count=self.layout.question_count,
                    options_per_question=self.layout.options_per_question,
                    template_type="vidyakashi_25",
                )
            elif not has_dark_banner and self.layout.template_type in ("vidyakashi_official", "vidyakashi_100"):
                self.layout = OMRLayoutConfig(
                    question_count=self.layout.question_count,
                    options_per_question=self.layout.options_per_question,
                    template_type="standard",
                )
            target_tl = self.layout.markers.get("top_left", (70, 70))
            if target_tl[1] > 200:
                target_tl, target_tr, target_br, target_bl = (70, 70), (1584, 70), (1584, 2268), (70, 2268)
            else:
                target_tr = self.layout.markers["top_right"]
                target_br = self.layout.markers["bottom_right"]
                target_bl = self.layout.markers["bottom_left"]
            target_marker_pts = np.array([target_tl, target_tr, target_br, target_bl], dtype="float32")

            ordered_src = self.order_quad_points(marker_pts)
            top_dx = float(ordered_src[1][0] - ordered_src[0][0])
            top_dy = float(ordered_src[1][1] - ordered_src[0][1])
            if top_dx > 10.0:
                marker_angle = math.degrees(math.atan2(top_dy, top_dx))
                if abs(marker_angle) >= 1.2:
                    self._last_alignment_info["was_tilted"] = True
                    self._last_alignment_info["tilt_angle_deg"] = round(marker_angle, 2)

            matrix = cv2.getPerspectiveTransform(ordered_src, target_marker_pts)
            rectified = cv2.warpPerspective(
                oriented_img, matrix, (CANONICAL_WIDTH, CANONICAL_HEIGHT), flags=cv2.INTER_LINEAR
            )
            if has_dark_banner:
                self.auto_calibrate_vidyakashi_grid(rectified)
            return rectified, True

        # 3. If dark Vidyakashi header banner is present without 4 black corner blocks -> Official 4-Column Vidyakashi Sheet
        if has_dark_banner:
            if self.layout.template_type not in ("vidyakashi_official", "vidyakashi_100"):
                self.layout = OMRLayoutConfig(
                    question_count=self.layout.question_count,
                    options_per_question=self.layout.options_per_question,
                    template_type="vidyakashi_official",
                )
            boundary_pts = self.detect_sheet_boundary(oriented_img)
            if boundary_pts is not None:
                page_target = np.array([
                    [0, 0],
                    [CANONICAL_WIDTH, 0],
                    [CANONICAL_WIDTH, CANONICAL_HEIGHT],
                    [0, CANONICAL_HEIGHT],
                ], dtype="float32")
                matrix = cv2.getPerspectiveTransform(boundary_pts, page_target)
                rectified = cv2.warpPerspective(
                    oriented_img, matrix, (CANONICAL_WIDTH, CANONICAL_HEIGHT), flags=cv2.INTER_LINEAR
                )
                self.auto_calibrate_vidyakashi_grid(rectified)
                return rectified, True

            resized = cv2.resize(oriented_img, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
            return resized, True

        # 3. Try sheet boundary detection for standard sheets
        boundary_pts = self.detect_sheet_boundary(oriented_img)
        if boundary_pts is not None:
            page_target = np.array([
                [0, 0],
                [CANONICAL_WIDTH, 0],
                [CANONICAL_WIDTH, CANONICAL_HEIGHT],
                [0, CANONICAL_HEIGHT],
            ], dtype="float32")
            matrix = cv2.getPerspectiveTransform(boundary_pts, page_target)
            rectified = cv2.warpPerspective(
                oriented_img, matrix, (CANONICAL_WIDTH, CANONICAL_HEIGHT), flags=cv2.INTER_LINEAR
            )
            return rectified, True

        # 4. Fallback: Direct resize
        resized = cv2.resize(oriented_img, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)
        return resized, False

    def validate_sheet_alignment_and_template(
        self,
        rectified_img: np.ndarray,
        markers_detected: bool,
    ) -> Dict[str, Any]:
        """
        Verifies:
        1. Whether the uploaded sheet layout matches the selected examination's OMR template.
        2. Whether all required answer bubble regions remain visible and uncropped after alignment.
        """
        gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY) if len(rectified_img.shape) == 3 else rectified_img
        h, w = gray.shape[:2]

        template_matches = True
        regions_visible = True
        validation_error: Optional[str] = None

        # 1. Check if layout has enough question bubbles for the selected examination
        detected_q_capacity = len(self.layout.bubbles)
        if detected_q_capacity < self.expected_question_count:
            template_matches = False
            validation_error = "The sheet layout does not match the selected examination."
        elif not markers_detected:
            banner_crop = gray[int(h * 0.02):int(h * 0.14), int(w * 0.15):int(w * 0.85)]
            has_vidyakashi_banner = bool(banner_crop.size > 0 and float(np.mean(banner_crop)) < 155.0)
            if has_vidyakashi_banner:
                regions_visible = False
                validation_error = "Part of the answer sheet is missing. Upload the complete sheet."
            else:
                # Verify whether circular OMR bubbles exist at expected locations
                visible_bubble_rows = 0
                for q_num, opt_map in self.layout.bubbles.items():
                    xs = [b.cx for b in opt_map.values()]
                    ys = [b.cy for b in opt_map.values()]
                    r = next(iter(opt_map.values())).radius
                    x1, x2 = max(0, min(xs) - r - 4), min(w, max(xs) + r + 4)
                    y1, y2 = max(0, min(ys) - r - 4), min(h, max(ys) + r + 4)
                    patch = gray[y1:y2, x1:x2]
                    if patch.size > 0 and float(np.std(patch)) >= 8.0:
                        visible_bubble_rows += 1
                if visible_bubble_rows < max(5, int(len(self.layout.bubbles) * 0.5)):
                    template_matches = False
                    validation_error = "The sheet layout does not match the selected examination."

        # 2. Check if all required answer regions (1..expected_question_count) are visible and uncropped
        if template_matches:
            missing_questions_list: List[int] = []
            for q_num in range(1, min(self.expected_question_count, len(self.layout.bubbles)) + 1):
                opt_map = self.layout.bubbles.get(q_num, {})
                if not opt_map:
                    missing_questions_list.append(q_num)
                    continue
                xs = [b.cx for b in opt_map.values()]
                ys = [b.cy for b in opt_map.values()]
                r = next(iter(opt_map.values())).radius
                if min(xs) - r < 2 or max(xs) + r >= w - 2 or min(ys) - r < 2 or max(ys) + r >= h - 2:
                    missing_questions_list.append(q_num)
                    continue
                x1, x2 = max(0, min(xs) - r - 3), min(w, max(xs) + r + 3)
                y1, y2 = max(0, min(ys) - r - 3), min(h, max(ys) + r + 3)
                patch = gray[y1:y2, x1:x2]
                # Even a blank OMR row has printed circle outlines (ptp > 20, std > 4.2)
                if patch.size == 0 or (float(np.max(patch)) - float(np.min(patch))) < 20.0 or float(np.std(patch)) < 4.2:
                    missing_questions_list.append(q_num)

            missing_questions = len(missing_questions_list)
            if missing_questions > max(2, int(self.expected_question_count * 0.10)):
                # Distinguish between template mismatch (e.g. Q1..25 present, Q26..100 unprinted) vs cropped sheet
                q1_to_25_missing = sum(1 for q in missing_questions_list if q <= 25)
                q26_plus_missing = sum(1 for q in missing_questions_list if q > 25)
                if self.expected_question_count > 25 and q1_to_25_missing <= 2 and q26_plus_missing >= int((self.expected_question_count - 25) * 0.6):
                    template_matches = False
                    validation_error = "The sheet layout does not match the selected examination."
                else:
                    regions_visible = False
                    validation_error = "Part of the answer sheet is missing. Upload the complete sheet."

        was_tilted = bool(self._last_alignment_info.get("was_tilted", False))
        tilt_deg = float(self._last_alignment_info.get("tilt_angle_deg", 0.0))
        rot_deg = int(self._last_alignment_info.get("rotation_deg", 0))

        alignment_message = (
            validation_error
            if validation_error
            else (
                "The OMR sheet is tilted. Aligning the image..."
                if was_tilted
                else "OMR sheet boundaries and template verified."
            )
        )

        return {
            "template_matches": template_matches,
            "regions_visible": regions_visible,
            "was_tilted": was_tilted,
            "tilt_angle_deg": tilt_deg,
            "rotation_deg": rot_deg,
            "validation_error": validation_error,
            "alignment_message": alignment_message,
        }

    def evaluate_bubble_features(
        self,
        gray_img: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> Dict[str, Any]:
        """
        Extracts multi-feature ink measurements inside the inner bubble region.
        Samples inner 50% radius to guarantee complete physical exclusion of printed circular borders.
        Measures:
        - Local background intensity (lighting, shadow, and camera glare invariant)
        - Mean inner ink contrast
        - Core center ink contrast (inner 28% radius, far removed from any border outline)
        - Dark ink pixel ratio (contrast >= 0.22)
        - Very dark ink pixel ratio (contrast >= 0.40)
        - Paper white pixel ratio (contrast < 0.12)
        - Core dark pixel ratio
        - Connected Component Analysis (CCA) for dots, pen strokes, and partial marks
        - Status classification: 'FILLED', 'EMPTY', or 'UNCERTAIN'
        """
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

        h_img, w_img = gray_img.shape[:2]
        inner_r = max(4, int(round(radius * 0.50)))
        core_r = max(2, int(round(radius * 0.28)))

        # 1. Local background estimation (92nd percentile in surrounding 1.5x box)
        box_half = max(inner_r + 6, int(radius * 1.5))
        bx1 = max(0, cx - box_half)
        bx2 = min(w_img, cx + box_half)
        by1 = max(0, cy - box_half)
        by2 = min(h_img, cy + box_half)

        roi_box = gray_img[by1:by2, bx1:bx2]
        bg_intensity = float(np.percentile(roi_box, 92)) if roi_box.size > 0 else 255.0
        if bg_intensity <= 20.0:
            bg_intensity = 255.0

        # 2. Extract inner circular ROI
        x1 = max(0, cx - inner_r)
        x2 = min(w_img, cx + inner_r + 1)
        y1 = max(0, cy - inner_r)
        y2 = min(h_img, cy + inner_r + 1)

        roi_inner = gray_img[y1:y2, x1:x2]
        if roi_inner.size == 0:
            return {
                "status": "EMPTY",
                "mean_contrast": 0.0,
                "core_contrast": 0.0,
                "dark_pixel_ratio": 0.0,
                "very_dark_ratio": 0.0,
                "core_dark_ratio": 0.0,
                "white_pixel_ratio": 1.0,
                "max_cc_area": 0,
                "total_ink_px": 0,
                "centroid": None,
                "inner_radius": inner_r,
                "total_inner_pixels": 0,
            }

        mask = np.zeros(roi_inner.shape, dtype=np.uint8)
        cv2.circle(mask, (cx - x1, cy - y1), inner_r, 255, -1)
        inner_pixels = roi_inner[mask > 0]
        total_inner = len(inner_pixels)
        if total_inner == 0:
            return {
                "status": "EMPTY",
                "mean_contrast": 0.0,
                "core_contrast": 0.0,
                "dark_pixel_ratio": 0.0,
                "very_dark_ratio": 0.0,
                "core_dark_ratio": 0.0,
                "white_pixel_ratio": 1.0,
                "max_cc_area": 0,
                "total_ink_px": 0,
                "centroid": None,
                "inner_radius": inner_r,
                "total_inner_pixels": 0,
            }

        core_mask = np.zeros(roi_inner.shape, dtype=np.uint8)
        cv2.circle(core_mask, (cx - x1, cy - y1), core_r, 255, -1)
        total_core = max(1, int(np.sum(core_mask > 0)))

        # 3. Normalized contrast relative to local paper background
        contrast_map = np.maximum(0.0, (bg_intensity - roi_inner.astype(np.float32)) / bg_intensity)
        mean_contrast = float(np.mean(contrast_map[mask > 0]))
        core_contrast = float(np.mean(contrast_map[core_mask > 0]))

        # Ink thresholding (calibrated for pen/pencil vs paper)
        dark_thresh = 0.18 if self.sensitivity == "HIGH" else 0.22 if self.sensitivity == "NORMAL" else 0.26
        dark_mask = (mask > 0) & (contrast_map >= dark_thresh)
        dark_pixel_ratio = float(np.sum(dark_mask) / float(total_inner))

        very_dark_mask = (mask > 0) & (contrast_map >= 0.40)
        very_dark_ratio = float(np.sum(very_dark_mask) / float(total_inner))

        core_dark_ratio = float(np.sum((core_mask > 0) & (contrast_map >= dark_thresh)) / float(total_core))
        white_pixel_ratio = float(np.sum((mask > 0) & (contrast_map < 0.12)) / float(total_inner))

        # 4. Connected Component Analysis for small dots / strokes
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask.astype(np.uint8), connectivity=8)
        max_cc_area = 0
        mark_centroid = None
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = int(np.argmax(areas)) + 1
            max_cc_area = int(stats[max_idx, cv2.CC_STAT_AREA])
            mark_centroid = (float(centroids[max_idx][0] + x1), float(centroids[max_idx][1] + y1))

        # 5. Tri-state classification of individual bubble: FILLED, EMPTY, UNCERTAIN
        # Outline immunity: if dark pixels only touch perimeter while core is empty and white ratio is high
        is_edge_artifact = bool(core_dark_ratio < 0.08 and white_pixel_ratio >= 0.65)
        is_empty_clean = bool(dark_pixel_ratio < 0.16 and core_dark_ratio < 0.16 and mean_contrast < 0.16)
        is_empty_letter = bool(white_pixel_ratio >= 0.30 and mean_contrast < 0.46 and very_dark_ratio < 0.55)

        is_filled_solid = bool(dark_pixel_ratio >= 0.55 and core_dark_ratio >= 0.48 and mean_contrast >= 0.32 and white_pixel_ratio <= 0.25)
        is_filled_letter = bool(dark_pixel_ratio >= 0.72 and core_dark_ratio >= 0.65 and mean_contrast >= 0.48 and white_pixel_ratio <= 0.18)
        is_filled_partial = bool(core_dark_ratio >= 0.35 and max_cc_area >= 18 and core_contrast >= 0.30 and mean_contrast >= 0.20)

        if is_filled_solid or is_filled_letter or is_filled_partial:
            bubble_status = "FILLED"
        elif is_edge_artifact or is_empty_clean or is_empty_letter:
            bubble_status = "EMPTY"
        elif mean_contrast >= 0.18:
            bubble_status = "UNCERTAIN"
        else:
            bubble_status = "EMPTY"

        return {
            "status": bubble_status,
            "mean_contrast": round(float(np.clip(mean_contrast, 0.0, 1.0)), 4),
            "core_contrast": round(float(np.clip(core_contrast, 0.0, 1.0)), 4),
            "dark_pixel_ratio": round(float(dark_pixel_ratio), 4),
            "very_dark_ratio": round(float(very_dark_ratio), 4),
            "core_dark_ratio": round(float(core_dark_ratio), 4),
            "white_pixel_ratio": round(float(white_pixel_ratio), 4),
            "max_cc_area": max_cc_area,
            "total_ink_px": int(np.sum(dark_mask)),
            "centroid": mark_centroid,
            "inner_radius": inner_r,
            "total_inner_pixels": total_inner,
        }

    def measure_bubble_ink_density(
        self, gray_img: np.ndarray, cx: int, cy: int, radius: int
    ) -> float:
        """
        Legacy contrast measurement.
        Returns the mean contrast inside the inner bubble region.
        """
        feats = self.evaluate_bubble_features(gray_img, cx, cy, radius)
        return feats["mean_contrast"]

    def decode_digit_grid(
        self,
        gray_img: np.ndarray,
        bubbles_list: List[Any],
        digit_count: int,
    ) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
        """
        Decodes a multi-column digit bubble grid (e.g. 7-digit Roll No., 7-digit Booklet No., 2-digit State).
        Accurately identifies:
        - Fully darkened bubbles in each column
        - Preserves leading zeros (e.g. '0123456')
        - Detects multiple darkened bubbles in the same column (marked as '*')
        - Detects missing/empty columns (marked as '?')
        - Detects faint/partial marks (marked as '?')
        Returns (digit_string, confidence_0_to_1, column_details).
        """
        if not bubbles_list or digit_count <= 0:
            return None, 0.0, []

        by_col: Dict[int, List[Any]] = {}
        for rb in bubbles_list:
            by_col.setdefault(rb.column_index, []).append(rb)

        col_digits = []
        confidences = []
        col_details = []

        for col_idx in range(digit_count):
            bubbles = by_col.get(col_idx, [])
            if not bubbles:
                col_digits.append("?")
                confidences.append(0.0)
                col_details.append({"status": "MISSING", "digit": "?", "conf": 0.0})
                continue

            evaluated = []
            for b in bubbles:
                feats = self.evaluate_bubble_features(gray_img, b.cx, b.cy, b.radius)
                evaluated.append((b.digit, feats))

            # Sort by mean_contrast descending
            evaluated.sort(key=lambda x: x[1]["mean_contrast"], reverse=True)

            # Check which bubbles in this column qualify as marked
            marked_candidates = []
            for d, feats in evaluated:
                status = feats.get("status", "EMPTY")
                mc = feats.get("mean_contrast", 0.0)
                dpr = feats.get("dark_pixel_ratio", 0.0)
                cdr = feats.get("core_dark_ratio", 0.0)
                if status == "FILLED" or (dpr >= 0.38 and mc >= 0.25) or (cdr >= 0.35 and mc >= 0.20):
                    marked_candidates.append((d, feats))

            if len(marked_candidates) == 0:
                # No bubble is filled
                col_digits.append("?")
                confidences.append(0.0)
                col_details.append({"status": "EMPTY", "digit": "?", "conf": 0.0})
            elif len(marked_candidates) == 1:
                # Exactly one bubble is filled
                lead_d, lead_f = marked_candidates[0]
                # Margin against second highest contrast bubble
                other_mc = [f["mean_contrast"] for d, f in evaluated if d != lead_d]
                runner_up_mc = max(other_mc) if other_mc else 0.0
                margin = lead_f["mean_contrast"] - runner_up_mc

                if margin >= 0.10 or lead_f["dark_pixel_ratio"] >= 0.48:
                    conf = min(1.0, max(0.85, 0.85 + margin))
                    col_digits.append(str(lead_d))
                    confidences.append(conf)
                    col_details.append({"status": "VALID", "digit": str(lead_d), "conf": round(conf, 2)})
                else:
                    col_digits.append("?")
                    confidences.append(0.40)
                    col_details.append({"status": "UNCERTAIN", "digit": "?", "conf": 0.40})
            else:
                # Multiple filled bubbles in the same column
                col_digits.append("*")
                confidences.append(0.20)
                col_details.append({"status": "MULTIPLE", "digit": "*", "conf": 0.20})

        result_str = "".join(col_digits)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return result_str, round(avg_conf, 2), col_details

    def detect_roll_number(self, gray_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Detects roll number with leading zero preservation and strict validation."""
        if not self.layout.roll_bubbles:
            return None, 0.0
        roll_str, conf, _ = self.decode_digit_grid(
            gray_img, self.layout.roll_bubbles, self.layout.roll_digit_count
        )
        return roll_str, conf

    def detect_booklet_number(self, gray_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Detects 7-digit Main Test Booklet Number from bubbled matrix."""
        if not getattr(self.layout, "booklet_bubbles", None):
            return None, 0.0
        booklet_str, conf, _ = self.decode_digit_grid(
            gray_img, self.layout.booklet_bubbles, self.layout.booklet_digit_count
        )
        return booklet_str, conf

    def detect_invigilator_codes(self, gray_img: np.ndarray) -> Dict[str, Optional[str]]:
        """Detects State (2 cols), Taluk (2 cols), and Centre (2 cols) code grids."""
        codes = {"state": None, "taluk": None, "centre": None}
        if getattr(self.layout, "state_bubbles", None):
            s_str, _, _ = self.decode_digit_grid(gray_img, self.layout.state_bubbles, 2)
            codes["state"] = s_str
        if getattr(self.layout, "taluk_bubbles", None):
            t_str, _, _ = self.decode_digit_grid(gray_img, self.layout.taluk_bubbles, 2)
            codes["taluk"] = t_str
        if getattr(self.layout, "centre_bubbles", None):
            c_str, _, _ = self.decode_digit_grid(gray_img, self.layout.centre_bubbles, 2)
            codes["centre"] = c_str
        return codes

    def detect_booklet_code(self, gray_img: np.ndarray) -> Optional[str]:
        """Detects Main Booklet Code (A, B, C, D)."""
        bubbles = getattr(self.layout, "booklet_code_bubbles", {})
        if not bubbles:
            return None
        marked = []
        for code, (cx, cy) in bubbles.items():
            feats = self.evaluate_bubble_features(gray_img, cx, cy, self.layout.bubble_radius)
            if feats["status"] == "FILLED" or (feats["dark_pixel_ratio"] >= 0.38 and feats["mean_contrast"] >= 0.25):
                marked.append((code, feats["mean_contrast"]))
        if len(marked) == 1:
            return marked[0][0]
        return None

    def detect_medium(self, gray_img: np.ndarray) -> Optional[str]:
        """Detects Examination Medium (English, Kannada)."""
        bubbles = getattr(self.layout, "medium_bubbles", {})
        if not bubbles:
            return None
        marked = []
        for med, (cx, cy) in bubbles.items():
            feats = self.evaluate_bubble_features(gray_img, cx, cy, self.layout.bubble_radius)
            if feats["status"] == "FILLED" or (feats["dark_pixel_ratio"] >= 0.38 and feats["mean_contrast"] >= 0.25):
                marked.append((med, feats["mean_contrast"]))
        if len(marked) == 1:
            return marked[0][0]
        return None

    def detect_vidyakashi_candidate_info(
        self, raw_or_rectified_img: np.ndarray
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Extracts Student Roll No. and Student Name from the candidate header box
        using cross-platform HTREngine with zero guessing and robust handwriting recognition.
        """
        try:
            tmpl_type = getattr(self.layout, "template_type", "vidyakashi_official")
            details = HTREngine.extract_candidate_details(
                canonical_img=raw_or_rectified_img,
                template_type=tmpl_type,
            )
            cand_roll = details.get("student_roll")
            cand_name = details.get("student_name")
            cand_conf = max(details.get("roll_confidence", 0.0), details.get("name_confidence", 0.0))
            return cand_roll, cand_name, cand_conf
        except Exception:
            return None, None, 0.0

    def detect_qr_code(self, img: np.ndarray) -> Optional[str]:
        """Attempts to read embedded QR code for sheet identifier verification."""
        try:
            detector = cv2.QRCodeDetector()
            val, pts, _ = detector.detectAndDecode(img)
            if val and val.strip():
                return val.strip()
        except Exception:
            pass
        return None

    def classify_question_bubbles(
        self,
        options_data: Dict[str, Dict[str, Any]],
        image_aligned: bool = True,
        is_blurred: bool = False,
    ) -> Tuple[str, float, Dict[str, bool]]:
        """
        Evaluates all option bubbles for a single question according to calibrated rules:
        - Exactly one bubble containing a detectable mark: classify that option as selected answer.
        - No detectable marks in any bubble: classify as BLANK.
        - Detectable marks in two or more bubbles: classify as MULTIPLE.
        - Image quality or ambiguous darkness: classify as REVIEW.
        Returns: (classification, confidence, mark_flags)
        """
        if not image_aligned or is_blurred:
            mark_flags = {opt: False for opt in options_data}
            return "REVIEW", 0.35, mark_flags

        all_contrasts = [d["mean_contrast"] for d in options_data.values()]
        all_dark_ratios = [d["dark_pixel_ratio"] for d in options_data.values()]
        min_contrast = min(all_contrasts)
        max_contrast = max(all_contrasts)
        min_dark_ratio = min(all_dark_ratios)

        # Detect if this template question has printed letters inside the bubbles
        # When printed letters are present, even empty bubbles have min_contrast >= 0.05 or min_dark_ratio >= 0.08
        has_letter_baseline = bool(min_contrast >= 0.05 or min_dark_ratio >= 0.08)

        mark_flags: Dict[str, bool] = {}

        for opt, data in options_data.items():
            status = data.get("status", "EMPTY")
            c = data["mean_contrast"]
            dr = data["dark_pixel_ratio"]
            wr = data.get("white_pixel_ratio", 1.0)
            cc = data.get("max_cc_area", 0)

            is_mark = False
            if has_letter_baseline:
                # To be marked over a printed letter baseline, must have substantial dark ink or stand out
                if (c - min_contrast >= 0.13 and dr >= 0.30) or (c - min_contrast >= 0.16 and (dr - min_dark_ratio >= 0.15)):
                    is_mark = True
                elif (c - min_contrast >= 0.16 or wr <= 0.20) and dr >= 0.50:
                    is_mark = True
                elif status == "FILLED" and wr <= 0.25 and (c - min_contrast >= 0.10):
                    is_mark = True
            else:
                # Clean circular bubbles (no printed letters inside)
                if status == "FILLED":
                    is_mark = True
                elif (c - min_contrast >= 0.14) and (dr - min_dark_ratio >= 0.14):
                    is_mark = True
                elif cc >= 8 and (dr - min_dark_ratio >= 0.02) and data.get("core_dark_ratio", 0.0) >= 0.04 and (c - min_contrast >= 0.02):
                    is_mark = True
                elif status == "UNCERTAIN" and (c - min_contrast >= 0.15):
                    is_mark = True
                else:
                    is_mark = False

            mark_flags[opt] = is_mark

        marked_opts = [opt for opt, flag in mark_flags.items() if flag]
        uncertain_opts = [opt for opt, d in options_data.items() if d.get("status") == "UNCERTAIN"]

        if len(marked_opts) == 0:
            if len(uncertain_opts) == 0:
                return "BLANK", 0.99, mark_flags
            # Uniform baseline check (e.g. all options have printed letters with similar contrast)
            if (max_contrast - min_contrast) < 0.14:
                return "BLANK", 0.98, mark_flags
            # Real ambiguous / faint mark: flag for teacher review
            return "REVIEW", 0.50, mark_flags

        elif len(marked_opts) == 1:
            selected = marked_opts[0]
            sel_c = options_data[selected]["mean_contrast"]
            other_c = [d["mean_contrast"] for o, d in options_data.items() if o != selected]
            runner_up = max(other_c) if other_c else 0.0

            # Verify contrast margin against runner-up
            if (sel_c - runner_up) >= 0.08 or options_data[selected]["dark_pixel_ratio"] >= 0.40 or options_data[selected].get("max_cc_area", 0) >= 8:
                confidence = round(min(1.0, max(0.75, 0.75 + (sel_c - runner_up))), 2)
                return selected, confidence, mark_flags
            return "REVIEW", 0.55, mark_flags

        else:
            sorted_marked = sorted(marked_opts, key=lambda o: options_data[o]["mean_contrast"], reverse=True)
            top_c = options_data[sorted_marked[0]]["mean_contrast"]
            sec_c = options_data[sorted_marked[1]]["mean_contrast"]

            # If one is heavily filled and second is an erasure (< 0.35) with large gap, flag REVIEW
            if top_c >= 0.65 and 0.18 <= sec_c < 0.35 and (top_c - sec_c) >= 0.35:
                return "REVIEW", 0.50, mark_flags

            return "MULTIPLE", 0.95, mark_flags

    def generate_visual_diagnostic(
        self,
        rectified_img: np.ndarray,
        questions: Dict[int, Dict[str, Any]],
        markers_detected: bool,
        answer_key: Optional[Dict[str, str]] = None,
        detected_roll: Optional[str] = None,
        detected_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Creates an annotated visual OMR correction & diagnostic image showing:
        - Registered sheet boundary & corner markers
        - Expected position and measured inner region of every bubble
        - Exact detected mark locations
        - When answer_key is provided:
          * CORRECT answers: Green circle + Green checkmark tag (e.g. "Q01:A OK")
          * WRONG answers: Red circle with Red X cross on student's bubble + Blue ring on the Correct Answer bubble + Red tag ("Q10:C X(B)")
          * BLANK answers: Orange/Blue ring on the Correct Answer bubble
        """
        diag = rectified_img.copy()

        # 1. Draw Corner Marker Expectations (Cyan boxes)
        for name, (cx, cy) in self.layout.markers.items():
            half = self.layout.marker_size // 2
            cv2.rectangle(
                diag,
                (cx - half, cy - half),
                (cx + half, cy + half),
                (255, 255, 0),  # Cyan
                2
            )
            cv2.putText(
                diag,
                name.replace("_", " ").upper(),
                (cx - half, cy - half - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1
            )

        correct_count = 0
        wrong_count = 0
        blank_count = 0
        multiple_count = 0

        # 2. Draw Every Question's Bubbles, Marks, and Correction vs Answer Key
        for q_num, opt_map in self.layout.bubbles.items():
            q_res = questions.get(q_num, {})
            classification = q_res.get("classification", "BLANK")
            densities = q_res.get("densities", {})
            opt_features = q_res.get("features", {})
            mark_flags = q_res.get("mark_flags", {})

            correct_opt = None
            if answer_key:
                raw_k = answer_key.get(str(q_num)) or answer_key.get(q_num)
                if raw_k:
                    correct_opt = str(raw_k).upper().strip()

            is_correct = False
            is_wrong = False
            if classification in ("A", "B", "C", "D"):
                if correct_opt in ("A", "B", "C", "D"):
                    if classification == correct_opt:
                        is_correct = True
                        correct_count += 1
                    else:
                        is_wrong = True
                        wrong_count += 1
                else:
                    is_correct = True
                    correct_count += 1
            elif classification == "BLANK":
                blank_count += 1
            else:
                multiple_count += 1
                if correct_opt in ("A", "B", "C", "D"):
                    wrong_count += 1

            for opt, bubble in opt_map.items():
                cx, cy, r = bubble.cx, bubble.cy, bubble.radius
                inner_r = max(3, int(r * 0.65))
                density = densities.get(opt, 0.0)
                b_feat = opt_features.get(opt, {})
                is_marked = mark_flags.get(opt, False)
                centroid = b_feat.get("centroid")

                # Bubble outline (thin circle)
                cv2.circle(diag, (cx, cy), r, (220, 180, 100), 1)
                cv2.circle(diag, (cx, cy), inner_r, (190, 190, 190), 1)

                if is_marked and centroid is not None:
                    mx, my = int(centroid[0]), int(centroid[1])
                    cv2.circle(diag, (mx, my), 3, (255, 255, 0), -1)

                # Color-code based on Answer Key correction
                if classification == opt:
                    if is_wrong:
                        # WRONG student answer: Red circle + Red X cross over bubble
                        cv2.circle(diag, (cx, cy), r + 3, (0, 0, 235), 3)
                        d = int(r * 0.65)
                        cv2.line(diag, (cx - d, cy - d), (cx + d, cy + d), (0, 0, 235), 3)
                        cv2.line(diag, (cx - d, cy + d), (cx + d, cy - d), (0, 0, 235), 3)
                    else:
                        # CORRECT student answer: Strong Green ring + center dot
                        cv2.circle(diag, (cx, cy), r + 3, (0, 190, 0), 3)
                        cv2.circle(diag, (cx, cy), 4, (0, 190, 0), -1)
                elif correct_opt == opt and (is_wrong or classification in ("BLANK", "MULTIPLE")):
                    # Show the actual correct option in Blue/Cyan ring when student got it wrong or left blank
                    cv2.circle(diag, (cx, cy), r + 3, (230, 110, 0), 3)
                elif classification == "MULTIPLE" and is_marked:
                    cv2.circle(diag, (cx, cy), r + 2, (0, 0, 255), 3)
                elif classification == "REVIEW" and (is_marked or density >= 0.15):
                    cv2.circle(diag, (cx, cy), r + 2, (0, 140, 255), 3)

            # Draw question correction label next to row
            first_b = next(iter(opt_map.values()))
            if is_wrong and correct_opt:
                tag_color = (0, 0, 230)  # Red for wrong
                q_tag = f"{classification} X (Key:{correct_opt})"
            elif is_correct:
                tag_color = (0, 155, 0)  # Green for correct
                q_tag = f"{classification} OK"
            elif classification == "BLANK" and correct_opt:
                tag_color = (180, 100, 0)
                q_tag = f"- (Key:{correct_opt})"
            else:
                tag_color = (0, 0, 220) if classification == "MULTIPLE" else (120, 120, 120)
                q_tag = f"{classification}"

            cv2.putText(
                diag,
                q_tag,
                (first_b.cx - 72, first_b.cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                tag_color,
                1
            )

        # 3. Top Banner Header Summary (OMR Correction Stamp)
        cv2.rectangle(diag, (0, 0), (CANONICAL_WIDTH, 48), (25, 30, 40), -1)
        total_q = len(self.layout.bubbles)
        cand_label = ""
        if detected_name or detected_roll:
            cand_label = f" | {detected_name or 'Student'} ({detected_roll or 'N/A'})"
        if answer_key:
            pct = round((correct_count / max(1, total_q)) * 100.0, 1)
            status_txt = f"OMR CORRECTION{cand_label} | SCORE: {correct_count}/{total_q} ({pct}%)"
            cv2.putText(diag, status_txt, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (100, 255, 140), 2)
            summary_txt = f"Correct: {correct_count} | Wrong: {wrong_count} | Blank: {blank_count}"
            cv2.putText(diag, summary_txt, (CANONICAL_WIDTH - 520, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        else:
            status_txt = f"OMR DIAGNOSTIC{cand_label} | Alignment: {'OK' if markers_detected else 'FALLBACK'}"
            cv2.putText(diag, status_txt, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            summary_txt = f"Detected: {correct_count} | Blank: {blank_count} | Multiple: {multiple_count}"
            cv2.putText(diag, summary_txt, (CANONICAL_WIDTH - 520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 230, 255), 1)

        return diag

    def process_omr_sheet(
        self,
        img: np.ndarray,
        answer_key: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete, calibrated OMR detection and correction pipeline on the image:
        1. Rectifies perspective and aligns with canonical geometry
        2. Detects blur / image quality
        3. Detects roll number (if applicable) and QR identifier
        4. Evaluates question bubbles with sensitive small-mark detection
        5. Classifies each question using calibrated decision rules
        6. Generates the visual OMR correction & diagnostic overlay (marking Right vs Wrong against answer_key)
        """
        rectified, markers_detected = self.rectify_image(img)
        alignment_info = self.validate_sheet_alignment_and_template(rectified, markers_detected)
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if len(rectified.shape) == 3 else rectified.copy()

        # Check image blur (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurred = lap_var < 35.0

        # 1. Student & Booklet Identification
        bubble_roll, bubble_conf = self.detect_roll_number(gray)
        booklet_number, booklet_conf = self.detect_booklet_number(gray)
        invig_codes = self.detect_invigilator_codes(gray)
        booklet_code = self.detect_booklet_code(gray)
        medium = self.detect_medium(gray)

        tmpl_type = getattr(self.layout, "template_type", "navodaya_80")
        htr_details = HTREngine.extract_candidate_details(
            canonical_img=rectified,
            template_type=tmpl_type,
            bubble_roll=bubble_roll,
        )
        detected_roll = bubble_roll if bubble_roll and "?" not in bubble_roll and "*" not in bubble_roll else (htr_details.get("student_roll") or bubble_roll)
        roll_conf = htr_details.get("roll_confidence", bubble_conf)
        detected_name = htr_details.get("student_name")
        name_conf = htr_details.get("name_confidence", 0.0)
        name_crop_url = htr_details.get("name_crop_url")
        father_name = htr_details.get("father_name")
        father_conf = htr_details.get("father_confidence", 0.0)
        father_crop_url = htr_details.get("father_crop_url")
        handwritten_roll = htr_details.get("handwritten_roll")
        handwritten_booklet = htr_details.get("handwritten_booklet")
        roll_match_status = htr_details.get("roll_match_status", "MATCH")
        details_need_review = (
            bool(htr_details.get("needs_review", False))
            or (not detected_roll)
            or ("?" in (detected_roll or ""))
            or ("*" in (detected_roll or ""))
            or (roll_match_status in ("MISMATCH", "VERIFICATION_REQUIRED"))
        )
        qr_identifier = self.detect_qr_code(rectified)

        # 2. Evaluate Question Bubbles
        question_results: Dict[int, Dict[str, Any]] = {}
        uncertain_count = 0

        for q_num, opt_map in self.layout.bubbles.items():
            options_data: Dict[str, Dict[str, Any]] = {}
            densities: Dict[str, float] = {}

            for opt, bubble in opt_map.items():
                b_feats = self.evaluate_bubble_features(gray, bubble.cx, bubble.cy, bubble.radius)
                options_data[opt] = b_feats
                densities[opt] = b_feats["mean_contrast"]

            classification, confidence, mark_flags = self.classify_question_bubbles(
                options_data=options_data,
                image_aligned=markers_detected,
                is_blurred=is_blurred,
            )

            if classification == "REVIEW":
                uncertain_count += 1

            question_results[q_num] = {
                "classification": classification,
                "confidence": confidence,
                "densities": densities,
                "features": options_data,
                "mark_flags": mark_flags,
            }

        # 3. Generate Visual Diagnostic / Correction Image
        diagnostic_img = self.generate_visual_diagnostic(
            rectified_img=rectified,
            questions=question_results,
            markers_detected=markers_detected,
            answer_key=answer_key,
            detected_roll=detected_roll,
            detected_name=detected_name,
        )

        return {
            "markers_detected": markers_detected,
            "alignment_info": alignment_info,
            "detected_roll_number": detected_roll,
            "detected_student_name": detected_name,
            "father_name": father_name,
            "father_confidence": father_conf,
            "father_crop_url": father_crop_url,
            "handwritten_roll": handwritten_roll,
            "handwritten_booklet": handwritten_booklet,
            "roll_match_status": roll_match_status,
            "booklet_number": booklet_number,
            "booklet_code": booklet_code,
            "medium": medium,
            "state_code": invig_codes.get("state"),
            "taluk_code": invig_codes.get("taluk"),
            "centre_code": invig_codes.get("centre"),
            "roll_confidence": roll_conf,
            "name_confidence": name_conf,
            "name_crop_url": name_crop_url,
            "details_need_review": details_need_review,
            "sheet_identifier": qr_identifier,
            "questions": question_results,
            "uncertain_questions_count": uncertain_count,
            "rectified_image": rectified,
            "diagnostic_image": diagnostic_img,
            "sensitivity": self.sensitivity,
            "is_blurred": is_blurred,
            "image_quality_score": round(lap_var, 1),
        }
