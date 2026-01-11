"""
Document Scanner Module
=======================

Robust computer vision module for automatic document detection
and perspective correction using OpenCV.

Handles low-contrast documents, shadows, and real-world backgrounds.

Author: Senior Computer Vision Engineer
Date: January 2026
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class DocumentScanner:
    """
    Main document scanner class providing document detection
    and perspective transformation.
    """

    def __init__(self):
        self.debug = False

    def scan_document(
        self,
        input_path: str,
        output_path: str,
        enhance_mode: str = "adaptive",
        width: int = 800
    ) -> Dict:

        try:
            original_image = cv2.imread(input_path)

            if original_image is None:
                return {
                    "success": False,
                    "message": "Could not read the uploaded image.",
                    "original_size": None,
                    "output_size": None
                }

            original_size = (original_image.shape[1], original_image.shape[0])

            # Preprocessing
            gray, ratio, processed = self.preprocess_image(original_image, width)

            # Document detection
            document_contour = self.detect_document_contour(processed)

            if document_contour is None:
                return {
                    "success": False,
                    "message": "Document boundary not detected. Please try better lighting.",
                    "original_size": original_size,
                    "output_size": None
                }

            # Scale contour back to original size
            document_contour = document_contour.reshape(4, 2) * ratio

            # Order points
            ordered_points = self.order_points(document_contour)

            # Perspective transform
            warped = self.four_point_transform(original_image, ordered_points)

            # Enhance scan
            enhanced = self.enhance_scan(warped, enhance_mode)

            # Save result
            cv2.imwrite(output_path, enhanced)

            return {
                "success": True,
                "message": "Document scanned successfully!",
                "original_size": original_size,
                "output_size": (enhanced.shape[1], enhanced.shape[0])
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error during scanning: {str(e)}",
                "original_size": None,
                "output_size": None
            }

    # ------------------------------------------------------------------

    def preprocess_image(
        self,
        image: np.ndarray,
        width: int
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Robust preprocessing using CLAHE + Adaptive Thresholding
        """

        ratio = image.shape[1] / float(width)
        dim = (width, int(image.shape[0] / ratio))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold (key fix)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # Morphological close to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.debug:
            cv2.imshow("Preprocessed", thresh)
            cv2.waitKey(0)

        return gray, ratio, thresh

    # ------------------------------------------------------------------

    def detect_document_contour(
        self,
        binary_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect document contour using area + polygon approximation
        """

        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        image_area = binary_image.shape[0] * binary_image.shape[1]

        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            if area < 0.2 * image_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                return approx

        # Fallback: min area rectangle
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return box.astype(int)

    # ------------------------------------------------------------------

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points as top-left, top-right, bottom-right, bottom-left
        """

        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # ------------------------------------------------------------------

    def four_point_transform(
        self,
        image: np.ndarray,
        pts: np.ndarray
    ) -> np.ndarray:
        """
        Perspective correction using homography
        """

        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # ------------------------------------------------------------------

    def enhance_scan(
        self,
        image: np.ndarray,
        mode: str
    ) -> np.ndarray:
        """
        Enhance final scanned output
        """

        if mode == "none":
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if mode == "adaptive":
            result = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        if mode == "clahe":
            clahe = cv2.createCLAHE(2.0, (8, 8))
            result = clahe.apply(gray)
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        if mode == "sharpen":
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])
            return cv2.filter2D(image, -1, kernel)

        return image


# -------------------------------------------------------------

def scan_document(input_path: str, output_path: str, enhance_mode: str = "adaptive") -> Dict:
    """
    Convenience wrapper
    """
    scanner = DocumentScanner()
    return scanner.scan_document(input_path, output_path, enhance_mode)
