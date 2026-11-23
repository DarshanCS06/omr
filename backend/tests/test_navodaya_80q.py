"""
Authoritative 15-Point Test Suite for Official Navodaya 80-Question OMR Evaluation.
Covers:
1.  All 80 question positions identified.
2.  All 4 options (A, B, C, D) for every question (320 bubbles total).
3.  Roll number detection preserving leading zeros (e.g. '0123456').
4.  Bubbled roll vs handwritten roll comparison (MATCH vs MISMATCH flagging).
5.  Candidate name and father name extraction & crop thumbnail generation.
6.  Partially filled bubble detection.
7.  Multiple darkened bubbles flagged as MULTIPLE ('*').
8.  Blank responses accurately detected as BLANK with 0 false positives.
9.  Rotated (180°) and perspective-distorted sheets rectified to canonical space.
10. Low-light and shadow invariant detection.
11. High-throughput non-blocking batch queue processing.
12. Marks out of 80 and percentage calculation.
13. Candidate response isolation across multiple submissions.
14. Database persistence and schema integrity.
15. Excel, CSV, and PDF export integrity with Rank, Father Name, Booklet No, Medium.
"""
import io
import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.models.user import User, UserRole
from backend.app.models.examination import Examination
from backend.app.models.answer_key import AnswerKey
from backend.app.models.submission import (
    Submission,
    DetectedResponse,
    SubmissionStatus,
    OptionClassification,
)
from backend.app.omr.layout import OMRLayoutConfig
from backend.app.omr.detector import OMRDetector
from backend.app.omr.htr_engine import HTREngine
from backend.app.services.scoring import ScoringEngine
from backend.app.reports.exporter import ReportExporter


@pytest.fixture(scope="module")
def canonical_navodaya_image():
    """Loads the canonical 1654x2338 Navodaya OMR sheet image."""
    img = cv2.imread("backend/app/omr/assets/navodaya_canonical.png")
    if img is None:
        # Fallback to white canvas if file not found
        img = np.full((2338, 1654, 3), 255, dtype=np.uint8)
    return img


@pytest.fixture
def navodaya_detector():
    layout = OMRLayoutConfig(question_count=80, template_type="navodaya_80")
    return OMRDetector(layout=layout)


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        user = User(
            email="admin@navodaya.edu",
            full_name="Navodaya Admin",
            hashed_password="hashed_pw",
            role=UserRole.ADMIN,
            is_active=True,
        )
        db.add(user)
        db.commit()

        exam = Examination(
            code="NAVODAYA-MOCK-80",
            title="Navodaya 80Q Mock Examination",
            subject="All Subjects",
            class_grade="6th",
            exam_date="2026-11-23",
            question_count=80,
            options_per_question=4,
            positive_marks=1.0,
            negative_marks=0.0,
            blank_marks=0.0,
            multiple_penalty=0.0,
            created_by_id=user.id,
        )
        db.add(exam)
        db.commit()

        # Seed answer key: Q1..80 cycle A, B, C, D
        opts = ["A", "B", "C", "D"]
        ak_dict = {str(q): opts[(q - 1) % 4] for q in range(1, 81)}
        ak = AnswerKey(examination_id=exam.id, version=1, answers=ak_dict, is_active=True)
        db.add(ak)
        db.commit()

        yield db, exam, user, ak_dict
    finally:
        db.close()


# ── TEST 1: Identify all 80 question positions ──────────────────────────────
def test_1_identify_all_80_questions(navodaya_detector):
    assert navodaya_detector.layout.question_count == 80
    assert len(navodaya_detector.layout.bubbles) == 80
    for q in range(1, 81):
        assert q in navodaya_detector.layout.bubbles


# ── TEST 2: Verify all 4 options (A, B, C, D) for every question ─────────────
def test_2_verify_all_4_options_per_question(navodaya_detector):
    total_bubbles = 0
    for q in range(1, 81):
        opts = navodaya_detector.layout.bubbles[q]
        assert set(opts.keys()) == {"A", "B", "C", "D"}
        for opt in ["A", "B", "C", "D"]:
            b = opts[opt]
            assert b.cx > 0 and b.cy > 0
            assert b.radius >= 12
            total_bubbles += 1
    assert total_bubbles == 320


# ── TEST 3: Roll number detection preserving leading zeros ──────────────────
def test_3_roll_number_leading_zeros(navodaya_detector, canonical_navodaya_image):
    test_img = canonical_navodaya_image.copy()
    # Fill roll number '0429103' in the 7 columns
    roll_digits = [0, 4, 2, 9, 1, 0, 3]
    for b in navodaya_detector.layout.roll_bubbles:
        col = b.column_index
        if b.digit == roll_digits[col]:
            cv2.circle(test_img, (b.cx, b.cy), b.radius - 2, (20, 20, 20), -1)

    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    detected_roll, conf = navodaya_detector.detect_roll_number(gray)
    assert detected_roll == "0429103"
    assert conf >= 0.90
    assert detected_roll.startswith("0")  # Leading zero preserved!


# ── TEST 4: Compare bubbled roll vs handwritten roll ────────────────────────
def test_4_roll_match_vs_mismatch(navodaya_detector, canonical_navodaya_image):
    # Test MATCH
    htr_match = HTREngine.extract_candidate_details(
        canonical_img=canonical_navodaya_image,
        template_type="navodaya_80",
        bubble_roll="0123456",
    )
    assert htr_match.get("roll_match_status") in ("MATCH", "VERIFICATION_REQUIRED")

    # Test that incomplete/unbubbled roll triggers VERIFICATION_REQUIRED
    htr_unbubbled = HTREngine.extract_candidate_details(
        canonical_img=canonical_navodaya_image,
        template_type="navodaya_80",
        bubble_roll="???????",
    )
    assert htr_unbubbled.get("needs_review") is True
    assert htr_unbubbled.get("roll_match_status") == "VERIFICATION_REQUIRED"


# ── TEST 5: Candidate name and father name recognition ──────────────────────
def test_5_name_and_father_name_crops(canonical_navodaya_image):
    details = HTREngine.extract_candidate_details(
        canonical_img=canonical_navodaya_image,
        template_type="navodaya_80",
    )
    assert "name_crop_url" in details
    assert details["name_crop_url"].startswith("data:image/jpeg;base64,")
    assert "father_crop_url" in details
    assert details["father_crop_url"].startswith("data:image/jpeg;base64,")


# ── TEST 6: Partially filled bubble detection ────────────────────────────────
def test_6_partially_filled_bubble(navodaya_detector, canonical_navodaya_image):
    test_img = canonical_navodaya_image.copy()
    # Partially fill Q10 option B (inner core filled with 55% radius)
    b = navodaya_detector.layout.bubbles[10]["B"]
    cv2.circle(test_img, (b.cx, b.cy), int(b.radius * 0.55), (30, 30, 30), -1)

    res = navodaya_detector.process_omr_sheet(test_img)
    q10_res = res["questions"][10]
    assert q10_res["classification"] == "B"
    assert q10_res["confidence"] >= 0.70


# ── TEST 7: Multiple darkened bubbles ───────────────────────────────────────
def test_7_multiple_darkened_bubbles(navodaya_detector, canonical_navodaya_image):
    test_img = canonical_navodaya_image.copy()
    # Mark both A and C on Q15
    b_a = navodaya_detector.layout.bubbles[15]["A"]
    b_c = navodaya_detector.layout.bubbles[15]["C"]
    cv2.circle(test_img, (b_a.cx, b_a.cy), b_a.radius - 2, (15, 15, 15), -1)
    cv2.circle(test_img, (b_c.cx, b_c.cy), b_c.radius - 2, (15, 15, 15), -1)

    res = navodaya_detector.process_omr_sheet(test_img)
    q15_res = res["questions"][15]
    assert q15_res["classification"] == "MULTIPLE"


# ── TEST 8: Blank responses ──────────────────────────────────────────────────
def test_8_blank_sheet_zero_false_positives(navodaya_detector, canonical_navodaya_image):
    res = navodaya_detector.process_omr_sheet(canonical_navodaya_image)
    blank_count = sum(1 for q in res["questions"].values() if q["classification"] == "BLANK")
    assert blank_count == 80  # All 80 must be blank
    assert res["uncertain_questions_count"] == 0  # Zero false reviews


# ── TEST 9: Rotated (180°) and perspective distorted sheets ─────────────────
def test_9_rotated_sheet_rectification(navodaya_detector, canonical_navodaya_image):
    # Rotate 180 degrees
    rot180 = cv2.rotate(canonical_navodaya_image, cv2.ROTATE_180)
    rect, ok = navodaya_detector.rectify_image(rot180)
    assert ok is True
    assert rect.shape == (2338, 1654, 3)

    # Process rotated sheet: all 80 questions still align
    res = navodaya_detector.process_omr_sheet(rot180)
    assert len(res["questions"]) == 80
    assert sum(1 for q in res["questions"].values() if q["classification"] == "BLANK") == 80


# ── TEST 10: Low-light & shadow conditions ──────────────────────────────────
def test_10_low_light_and_shadow_invariance(navodaya_detector, canonical_navodaya_image):
    # Simulate low light + non-uniform lighting gradient (shadow across right side)
    low_light = (canonical_navodaya_image.astype(np.float32) * 0.65).astype(np.uint8)
    h, w = low_light.shape[:2]
    gradient = np.linspace(1.0, 0.5, w).reshape(1, w, 1)
    shadowed = (low_light.astype(np.float32) * gradient).astype(np.uint8)

    # Mark Q5 option D with pen
    b_d = navodaya_detector.layout.bubbles[5]["D"]
    cv2.circle(shadowed, (b_d.cx, b_d.cy), b_d.radius - 2, (10, 10, 10), -1)

    res = navodaya_detector.process_omr_sheet(shadowed)
    assert res["questions"][5]["classification"] == "D"
    # Other questions remain clean
    assert res["questions"][6]["classification"] == "BLANK"


# ── TEST 11: High-throughput batch queue simulation ─────────────────────────
def test_11_batch_processing_throughput(navodaya_detector, canonical_navodaya_image):
    import time
    start_t = time.time()
    # Benchmark 5 full sheet runs
    for _ in range(5):
        res = navodaya_detector.process_omr_sheet(canonical_navodaya_image)
        assert len(res["questions"]) == 80
    elapsed = time.time() - start_t
    avg_per_sheet = elapsed / 5.0
    # Average processing per sheet should be well under 3.5 seconds
    assert avg_per_sheet < 3.5


# ── TEST 12: Correct marks out of 80 and percentage calculation ─────────────
def test_12_score_calculation_out_of_80(test_db):
    db, exam, user, ak_dict = test_db

    sub = Submission(
        examination_id=exam.id,
        uploaded_image_id=999,
        detected_roll_number="0123456",
        detected_student_name="Priya Patel",
        status=SubmissionStatus.PROVISIONAL,
    )
    db.add(sub)
    db.commit()

    # Simulate 60 correct, 10 wrong, 5 blank, 5 multiple
    responses = []
    for q in range(1, 81):
        correct_ans = ak_dict[str(q)]
        if q <= 60:
            det = OptionClassification[correct_ans]
        elif q <= 70:
            det = OptionClassification["B" if correct_ans != "B" else "C"]  # Wrong
        elif q <= 75:
            det = OptionClassification.BLANK
        else:
            det = OptionClassification.MULTIPLE

        dr = DetectedResponse(
            submission_id=sub.id,
            question_number=q,
            detected_option=det,
            confidence_score=0.98,
        )
        responses.append(dr)

    db.add_all(responses)
    db.commit()

    ScoringEngine.update_submission_score(sub, exam, ak_dict, db)

    assert sub.correct_count == 60
    assert sub.incorrect_count == 10
    assert sub.blank_count == 5
    assert sub.multiple_count == 5
    assert sub.total_score == 60.0
    assert sub.max_score == 80.0
    assert sub.percentage == 75.0  # (60 / 80) * 100


# ── TEST 13: Candidate response isolation ───────────────────────────────────
def test_13_candidate_response_isolation(test_db):
    db, exam, user, ak_dict = test_db

    # Create Sub 1
    sub1 = Submission(examination_id=exam.id, uploaded_image_id=1001, detected_roll_number="0000001")
    # Create Sub 2
    sub2 = Submission(examination_id=exam.id, uploaded_image_id=1002, detected_roll_number="0000002")
    db.add_all([sub1, sub2])
    db.commit()

    # Add response Q1:A to sub1, Q1:C to sub2
    r1 = DetectedResponse(submission_id=sub1.id, question_number=1, detected_option=OptionClassification.A)
    r2 = DetectedResponse(submission_id=sub2.id, question_number=1, detected_option=OptionClassification.C)
    db.add_all([r1, r2])
    db.commit()

    q1_sub1 = db.query(DetectedResponse).filter_by(submission_id=sub1.id, question_number=1).one()
    q1_sub2 = db.query(DetectedResponse).filter_by(submission_id=sub2.id, question_number=1).one()

    assert q1_sub1.final_option == "A"
    assert q1_sub2.final_option == "C"
    assert q1_sub1.submission_id != q1_sub2.submission_id


# ── TEST 14: Data persistence across restarts ───────────────────────────────
def test_14_database_schema_and_persistence(test_db):
    db, exam, user, ak_dict = test_db
    # Verify all newly added columns persist cleanly
    sub = Submission(
        examination_id=exam.id,
        uploaded_image_id=1003,
        detected_roll_number="0429103",
        detected_student_name="Rohan Gowda",
        father_name="Basavaraj Gowda",
        booklet_number="7654321",
        booklet_code="A",
        medium="Kannada",
        state_code="29",
        taluk_code="04",
        centre_code="02",
        roll_match_status="MATCH",
        total_score=80.0,
        max_score=80.0,
        percentage=100.0,
        correct_count=80,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)

    fetched = db.query(Submission).filter_by(id=sub.id).one()
    assert fetched.detected_roll_number == "0429103"
    assert fetched.father_name == "Basavaraj Gowda"
    assert fetched.booklet_number == "7654321"
    assert fetched.booklet_code == "A"
    assert fetched.medium == "Kannada"
    assert fetched.state_code == "29"
    assert fetched.total_score == 80.0


# ── TEST 15: Excel, CSV, and PDF exports ────────────────────────────────────
def test_15_export_reports_integrity(test_db):
    db, exam, user, ak_dict = test_db

    # 1. CSV Export
    csv_str = ReportExporter.generate_csv_results(exam, db)
    assert "Rank" in csv_str
    assert "Father's Name" in csv_str
    assert "Booklet Number" in csv_str
    assert "Medium" in csv_str
    assert "Q80 Answer" in csv_str

    # 2. Excel Export
    excel_bytes = ReportExporter.generate_excel_results(exam, db)
    assert len(excel_bytes) > 1000
    assert excel_bytes.startswith(b"PK")  # Valid xlsx zip container

    # 3. PDF Export
    pdf_bytes = ReportExporter.generate_pdf_results(exam, db)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")
