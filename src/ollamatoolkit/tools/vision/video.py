# ./ollamatoolkit/tools/vision/video.py
"""
Ollama Toolkit - Smart Video Processor
======================================
Advanced video handling with Scene Detection.
"""

import os
import tempfile
from typing import List
import cv2


class VideoProcessor:
    def __init__(self):
        pass

    def extract_keyframes(
        self, video_path: str, min_interval_sec: float = 1.0, threshold: float = 0.6
    ) -> List[str]:
        """
        Extracts representative frames based on visual content changes (Scene Detection).

        Args:
            video_path: Path to video.
            min_interval_sec: Minimum time between frames (debounce).
            threshold: Correlation threshold (lower = more sensitive to change).
                       0.6 is a good starting point for histogram comparison (bhattacharyya).

        Returns:
            List[str]: Paths to keyframes.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Output setup
        temp_dir = os.path.join(tempfile.gettempdir(), "ollama_vision_video")
        os.makedirs(temp_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        min_frame_interval = int(fps * min_interval_sec)

        frame_paths = []
        last_hist = None
        frame_count = 0
        last_saved_frame = -min_frame_interval  # Force first frame

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 1. Check interval constraint
            if (frame_count - last_saved_frame) < min_frame_interval:
                frame_count += 1
                continue

            # 2. Calculate Histogram
            # HSV histogram is usually better for scene change than RGB
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            is_keyframe = False

            if last_hist is None:
                is_keyframe = True  # First frame always key
            else:
                # Compare Histograms
                # Correlation method: 1.0 = identical, 0.0 = opposite
                # Bhattacharyya: 0.0 = match, 1.0 = mismatch. Let's use Correlation (method 0)
                # score = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                # if score < threshold: is_keyframe = True

                # Let's use Bhattacharyya (method 3) for robustness?
                # Dist 0.0 (match) -> 1.0 (mismatch)
                score = cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if score > (
                    1.0 - threshold
                ):  # If diff > 0.4 (assuming thresh 0.6 means 0.6 match)
                    is_keyframe = True

            if is_keyframe:
                out_path = os.path.join(
                    temp_dir, f"{base_name}_scene_{frame_count}.jpg"
                )
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)
                last_saved_frame = frame_count
                last_hist = hist

            frame_count += 1

        cap.release()
        return frame_paths
