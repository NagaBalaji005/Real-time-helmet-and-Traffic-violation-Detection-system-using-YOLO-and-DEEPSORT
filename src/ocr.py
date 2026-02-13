#!/usr/bin/env python3
"""
FINAL Indian Number Plate OCR using PaddleOCR
- Integrates proven PaddleOCR preprocessing from plate_ocr_paddle.py
- Advanced preprocessing variations for cropped plate images
- Multi-strategy correction and validation
- Format: AA BB CC DDDD (State RTO Series Number)
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Tuple, Optional
import re

class UltimateOCR:
    """Production-grade OCR with PaddleOCR - Optimized for cropped plates"""
    
    def __init__(self, use_angle_cls=True, lang='en', use_gpu=False):
        """Initialize PaddleOCR with optimal settings"""
        # Initialize PaddleOCR - handle version differences gracefully
        # Newer versions don't support show_log, use_gpu, or some other parameters
        try:
            # Try with all parameters first (older PaddleOCR versions)
            self.reader = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=6
            )
        except (TypeError, ValueError) as e:
            # Fallback: try without show_log (newer versions)
            try:
                self.reader = PaddleOCR(
                    use_angle_cls=use_angle_cls,
                    lang=lang,
                    use_gpu=use_gpu,
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=6
                )
            except (TypeError, ValueError):
                # Fallback: try without use_gpu
                try:
                    self.reader = PaddleOCR(
                        use_angle_cls=use_angle_cls,
                        lang=lang,
                        det_db_thresh=0.3,
                        det_db_box_thresh=0.5
                    )
                except (TypeError, ValueError):
                    # Final fallback: minimal initialization
                    self.reader = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        
        # Valid Indian state codes
        self.valid_states = {
            'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 
            'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 
            'MH', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 
            'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
        }
        
        # Character remapping for OCR corrections (expanded from plate_ocr_paddle.py)
        # CRITICAL: Only remap when position suggests it should be digit
        # Do NOT remap E->9, J->3, G->6 in series positions (they are valid letters)
        self.char_map = {
            'O': '0', 'Q': '0',  # O and Q often confused with 0
            'I': '1', 'L': '1',  # I and L often confused with 1
            'Z': '2',  # Z confused with 2
            'S': '5',  # S confused with 5
            'B': '8',  # B confused with 8
            # DO NOT remap G->6, J->3, E->9 in series positions - they are valid letters!
            'T': '7',  # T confused with 7 (only in number positions)
        }
        
        # Reverse mapping for letters (digit to letter conversion)
        # CRITICAL: 6->G, 3->J, 9->E are common OCR errors in series positions
        self.reverse_char_map = {
            '0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S', 
            '6': 'G',  # 6 often misread as G in series
            '7': 'T', 
            '3': 'J',  # 3 often misread as J in series
            '9': 'E'   # 9 often misread as E in series
        }
        
        # RTO code fixes (common OCR errors)
        self.rto_fixes = {
            'OC': '09', 'OS': '05', 'OI': '01', 'OZ': '02', 'OB': '08',
            'CO': '90', 'SO': '50', 'IO': '10', 'ZO': '20', 'BO': '80',
            'C9': '09', 'S5': '55', 'OD': '00', 'DO': '00', 'OO': '00',
            'O0': '00', '0O': '09', 'O9': '09', 'OA': '04', 'AO': '40',
            'OG': '06', 'GO': '60', 'OT': '07', 'TO': '70'
        }
        
        # Plate format regex - prioritize 2-letter series
        # Part 4 (number) can be 1-4 digits (1-9999)
        self.plate_regex = r"([A-Z]{2})([0-9]{2})([A-Z]{2})([0-9]{1,4})"  # Prefer 2-letter series, 1-4 digit number
        self.plate_regex_flexible = r"([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{1,4})"  # Fallback for 1-letter series, 1-4 digit number
        self.plate_regex_4digit = r"([A-Z]{2})([0-9]{2})([A-Z]{2})([0-9]{4})"  # Strict 4-digit for validation
        
        print("✅ PaddleOCR Ultimate OCR initialized")
    
    def read_plate(self, plate_image: np.ndarray, verbose: bool = False) -> Optional[Dict]:
        """
        Main OCR pipeline with PaddleOCR - optimized for cropped plates
        Uses preprocessing strategies from plate_ocr_paddle.py
        
        Args:
            plate_image: Cropped number plate image
            verbose: If True, print detailed debug information
        """
        if plate_image is None or plate_image.size == 0:
            return None
        
        h, w = plate_image.shape[:2]
        if verbose:
            print(f"\n{'='*80}")
            print(f"🔍 PADDLEOCR ULTIMATE OCR PIPELINE")
            print(f"{'='*80}")
            print(f"Input: {w}x{h} pixels")
        
        # STAGE 1: Generate preprocessing variations
        if verbose:
            print(f"\n📊 STAGE 1: Preprocessing")
        variations = self._generate_variations(plate_image, verbose=verbose)
        if verbose:
            print(f"✅ Generated {len(variations)} variations")
        
        # STAGE 2: Run OCR on all variations
        if verbose:
            print(f"\n📊 STAGE 2: OCR Extraction")
            print(f"{'─'*80}")
        
        best_result = None
        best_score = 0
        all_attempts = []
        
        for idx, (name, processed) in enumerate(variations, 1):
            try:
                # Run PaddleOCR
                results = self.reader.ocr(processed, cls=True)
                
                if not results or not results[0]:
                    if verbose:
                        print(f"[{idx:02d}] {name:35s} → No text")
                    continue
                
                # Combine OCR results
                texts = []
                confs = []
                for line in results[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.05:
                        clean = ''.join(c for c in text.upper() if c.isalnum())
                        if clean:
                            texts.append(clean)
                            confs.append(conf)
                
                if not texts:
                    if verbose:
                        print(f"[{idx:02d}] {name:35s} → Empty")
                    continue
                
                raw_text = ''.join(texts)
                avg_conf = sum(confs) / len(confs)
                
                # Calculate pattern score
                pattern_score = self._calculate_pattern_score(raw_text)
                
                # Try formatting with corrections - prioritize 2-letter series
                formatted = self._format_with_corrections(raw_text)
                
                if formatted:
                    # Extract parts and validate strictly
                    parts = self._extract_parts(formatted)
                    state_code = parts.get('state', '')
                    rto_code = parts.get('rto', '')
                    series_code = parts.get('series', '')
                    number_code = parts.get('number', '')
                    
                    # CRITICAL: Strict validation - all parts must be valid
                    if not state_code or state_code not in self.valid_states:
                        if verbose:
                            print(f"[{idx:02d}] {name:35s} → {formatted:15s} "
                                  f"(Conf: {avg_conf:.2%}, ❌ Invalid state: {state_code})")
                        continue
                    
                    if not rto_code or not rto_code.isdigit() or len(rto_code) != 2:
                        if verbose:
                            print(f"[{idx:02d}] {name:35s} → {formatted:15s} "
                                  f"(Conf: {avg_conf:.2%}, ❌ Invalid RTO: {rto_code})")
                        continue
                    
                    # Part 4: Number can be 1-4 digits (1-9999)
                    if not number_code or not number_code.isdigit() or len(number_code) < 1 or len(number_code) > 4:
                        if verbose:
                            print(f"[{idx:02d}] {name:35s} → {formatted:15s} "
                                  f"(Conf: {avg_conf:.2%}, ❌ Invalid number: {number_code} - must be 1-4 digits)")
                        continue
                    
                    # If 4 digits, cannot start with 0
                    if len(number_code) == 4 and number_code[0] == '0':
                        if verbose:
                            print(f"[{idx:02d}] {name:35s} → {formatted:15s} "
                                  f"(Conf: {avg_conf:.2%}, ❌ Invalid number: {number_code} - 4-digit cannot start with 0)")
                        continue
                    
                    # Series is optional but if present must be valid
                    if series_code and (not series_code.isalpha() or len(series_code) > 3):
                        if verbose:
                            print(f"[{idx:02d}] {name:35s} → {formatted:15s} "
                                  f"(Conf: {avg_conf:.2%}, ❌ Invalid series: {series_code})")
                        continue
                    
                    # CRITICAL: If raw text has 8+ alphanumeric chars, series MUST exist
                    # Reject plates without series if raw text suggests series should exist
                    raw_cleaned_len = len(re.sub(r'[^A-Z0-9]', '', raw_text.upper()))
                    if raw_cleaned_len >= 8 and not series_code:
                        # Raw text is long enough for series - likely missing series (like "KL 30 6392" should be "KL 30 G 6392")
                        if verbose:
                            print(f"[{idx:02d}] {name:35s} → {formatted:15s} "
                                  f"(Conf: {avg_conf:.2%}, ❌ Missing Part 3 (series) - raw length {raw_cleaned_len} requires series)")
                        continue
                    
                    # Prefer 2-letter series (most common format)
                    series_bonus = 100 if len(series_code) == 2 else 0
                    
                    # Calculate final score
                    final_score = (pattern_score * 1000) + (avg_conf * 10) + series_bonus
                    
                    if verbose:
                        print(f"[{idx:02d}] {name:35s} → {formatted:15s} "
                              f"(Raw: {raw_text:10s}, Conf: {avg_conf:.2%}, "
                              f"Pattern: {pattern_score:.1f}, Score: {final_score:.1f})")
                    
                    attempt = {
                        'method': name,
                        'raw_text': raw_text,
                        'formatted': formatted,
                        'confidence': avg_conf,
                        'pattern_score': pattern_score,
                        'final_score': final_score
                    }
                    all_attempts.append(attempt)
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_result = {
                            'text': formatted,
                            'confidence': avg_conf,
                            'score': self._calculate_quality_score(parts, avg_conf),
                            'is_valid': True,
                            'parts': parts,
                            'raw_text': raw_text,  # Store raw OCR text for debugging
                            'method': name,
                            'pattern_score': pattern_score
                        }
                else:
                    if verbose:
                        print(f"[{idx:02d}] {name:35s} → {raw_text:15s} "
                              f"(Conf: {avg_conf:.2%}, ❌ Could not format)")
            
            except Exception as e:
                if verbose:
                    print(f"[{idx:02d}] {name:35s} → ERROR: {str(e)[:30]}")
                continue
        
        # No fallback normalization - strict validation only
        
        # STAGE 3: Results
        if verbose:
            print(f"\n{'─'*80}")
            print(f"📊 STAGE 3: Final Results")
            print(f"{'='*80}")
            
            if best_result:
                print(f"✅ SUCCESS!")
                print(f"   Plate: {best_result['text']}")
                print(f"   Confidence: {best_result['confidence']:.2%}")
                print(f"   Quality Score: {best_result['score']:.2f}/15")
                print(f"   Pattern Score: {best_result['pattern_score']:.1f}")
                print(f"   Method: {best_result['method']}")
                print(f"   Parts: {best_result['parts']}")
                print(f"   Attempts: {len(all_attempts)} successful")
            else:
                print(f"❌ FAILED")
                if all_attempts:
                    print(f"   Best raw: {all_attempts[0]['raw_text']}")
            
            print(f"{'='*80}\n")
        
        return best_result
    
    def _generate_variations(self, img: np.ndarray, verbose: bool = False) -> List[Tuple[str, np.ndarray]]:
        """
        Generate preprocessing variations optimized for PaddleOCR
        Based on proven plate_ocr_paddle.py preprocessing
        """
        variations = []
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # CRITICAL: PaddleOCR benefits from strong upscaling (from plate_ocr_paddle.py)
        h, w = gray.shape
        if w < 200 or h < 60:
            scale = max(3.0, 200/w, 60/h)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
            if verbose:
                print(f"   ⚡ Upscaled {w}x{h} → {gray.shape[1]}x{gray.shape[0]}")
        
        # === GROUP 1: Proven plate_ocr_paddle.py preprocessing ===
        
        # V1: Core plate_ocr_paddle.py method
        v1_denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        v1_enhanced = clahe.apply(v1_denoised)
        variations.append(("PaddleCore_Denoised_CLAHE", v1_enhanced))
        
        # V2: Original without denoise
        v2_enhanced = clahe.apply(gray)
        variations.append(("PaddleCore_CLAHE_Only", v2_enhanced))
        
        # V3: CLAHE + Adaptive Threshold
        filtered = cv2.bilateralFilter(v1_enhanced, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        variations.append(("PaddleCore_CLAHE_Adaptive", thresh))
        
        # === GROUP 2: Enhanced variations ===
        
        # V4: Stronger CLAHE
        strong_clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        v4_enhanced = strong_clahe.apply(v1_denoised)
        variations.append(("Enhanced_Strong_CLAHE", v4_enhanced))
        
        # V5: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        v5_morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        variations.append(("Enhanced_Morph_Close", v5_morph))
        
        # V6: Otsu's thresholding
        _, v6_otsu = cv2.threshold(v1_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variations.append(("Enhanced_Otsu", v6_otsu))
        
        # === GROUP 3: Alternative preprocessing ===
        
        # V7: Multiple Gaussian blur + CLAHE
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        v7_enhanced = clahe.apply(blurred)
        variations.append(("Alternative_Gaussian_CLAHE", v7_enhanced))
        
        # V8: Unsharp masking
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        v8_unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        v8_enhanced = clahe.apply(v8_unsharp)
        variations.append(("Alternative_Unsharp_CLAHE", v8_enhanced))
        
        # === GROUP 4: Brightness adjustments (for varied lighting) ===
        
        for brightness in [0.8, 1.2]:
            adjusted = cv2.convertScaleAbs(gray, alpha=brightness, beta=0)
            adj_enhanced = clahe.apply(adjusted)
            variations.append((f"Brightness_{brightness}_CLAHE", adj_enhanced))
        
        # === GROUP 5: Rotation corrections (small angles) ===
        
        for angle in [-2, -1, 1, 2]:
            h_rot, w_rot = gray.shape
            M = cv2.getRotationMatrix2D((w_rot//2, h_rot//2), angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w_rot, h_rot), 
                                    borderMode=cv2.BORDER_REPLICATE)
            rot_enhanced = clahe.apply(rotated)
            variations.append((f"Rotation_{angle}deg_CLAHE", rot_enhanced))
        
        return variations
    
    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate how well text matches Indian plate pattern"""
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        score = 0.0
        
        # Perfect match: AA BB CC DDDD
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{0,3}\d{1,4}$', cleaned):
            score = 3.0
            if len(cleaned) >= 2 and cleaned[:2] in self.valid_states:
                score += 1.0
            if len(cleaned) >= 6:
                series_part = cleaned[4:6]
                if series_part and series_part[0].isalpha():
                    score += 0.5
        elif re.match(r'^[A-Z]{2}\d{2}', cleaned):
            score = 2.0
            if len(cleaned) >= 2 and cleaned[:2] in self.valid_states:
                score += 0.5
        elif 5 <= len(cleaned) <= 10:
            score = 1.0
        elif len(cleaned) >= 3:
            score = 0.5
        
        return score
    
    def _format_with_corrections(self, raw_text: str) -> Optional[str]:
        """Format plate with multiple correction strategies"""
        if not raw_text or len(raw_text) < 5:
            return None
        
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        
        if len(cleaned) < 6 or len(cleaned) > 12:  # Allow up to 12 chars for OCR errors
            return None
        
        # Strategy 1: Direct parsing
        result = self._parse_direct(cleaned)
        if result:
            return result
        
        # Strategy 2: Try removing single characters (for OCR errors like TS09ET9099 or BRO1C7F8627)
        if len(cleaned) >= 10:
            for remove_pos in range(4, len(cleaned) - 4):
                test_text = cleaned[:remove_pos] + cleaned[remove_pos+1:]
                result = self._parse_direct(test_text)
                if result:
                    return result
        
        # Strategy 3: Position-based corrections
        result = self._parse_position_corrected(cleaned)
        if result:
            return result
        
        # Strategy 4: RTO code fixes
        result = self._parse_rto_fixed(cleaned)
        if result:
            return result
        
        # Strategy 5: Comprehensive remapping
        result = self._parse_remapped(cleaned)
        if result:
            return result
        
        # Strategy 6: Reconstruct from regex pattern
        result = self._reconstruct_from_pattern(cleaned)
        if result:
            return result
        
        return None
    
    def _parse_direct(self, text: str) -> Optional[str]:
        """Try direct parsing (no corrections) - prioritize 2-letter series first"""
        # CRITICAL: Try 2-letter series FIRST (most common format like "UP 65 AE 7191")
        # This prevents misparsing "UP65AE7191" as "UP 65 A 9719"
        # Also handle 1-letter series like "KL 30 G 392" -> "KL30G392"
        for series_len in [2, 1, 3, 0]:  # 2-letter series first!
            if len(text) < 6 + series_len:
                continue
            
            state = text[:2]
            rto = text[2:4]
            series = text[4:4+series_len] if series_len > 0 else ''
            number = text[4+series_len:]
            
            # CRITICAL: Number must be 1-4 digits (not strictly 4) - can be 9, 99, 999, or 9999
            if not number.isdigit() or len(number) < 1 or len(number) > 4:
                continue
            
            # If 4 digits, cannot start with 0
            if len(number) == 4 and number[0] == '0':
                continue
            
            # Validate all parts - this will check series requirement based on raw_text_length
            if self._validate_parts(state, rto, series, number, raw_text_length=len(text)):
                return self._format_plate(state, rto, series, number)
        
        # Try with character removal for cases like TS09ET9099 (should parse as TS 09 ET 9099)
        # or BRO1C7F8627 (remove 7 to get BR 01 CF 8627)
        if len(text) >= 10:
            # Try removing single digits/characters that might be OCR errors
            for remove_pos in range(4, len(text) - 4):
                test_text = text[:remove_pos] + text[remove_pos+1:]
                # Try series lengths in order: 2, 1, 3, 0
                for series_len in [2, 1, 3, 0]:
                    if len(test_text) < 6 + series_len:
                        continue
                    state = test_text[:2]
                    rto = test_text[2:4]
                    series = test_text[4:4+series_len] if series_len > 0 else ''
                    number = test_text[4+series_len:]
                    
                    # CRITICAL: Number must be 1-4 digits (not strictly 4)
                    if not number.isdigit() or len(number) < 1 or len(number) > 4:
                        continue
                    
                    # If 4 digits, cannot start with 0
                    if len(number) == 4 and number[0] == '0':
                        continue
                    
                    if self._validate_parts(state, rto, series, number, raw_text_length=len(text)):
                        return self._format_plate(state, rto, series, number)
        
        return None
    
    def _parse_position_corrected(self, text: str) -> Optional[str]:
        """Position-based character corrections with smart remapping"""
        corrected = list(text)
        
        # First 2 positions: should be letters (state code)
        for i in range(min(2, len(corrected))):
            if corrected[i].isdigit():
                corrected[i] = self.reverse_char_map.get(corrected[i], corrected[i])
        
        # Positions 3-4: should be digits (RTO code)
        for i in range(2, min(4, len(corrected))):
            if corrected[i].isalpha():
                # Only remap if it's a commonly confused character
                if corrected[i] in ['O', 'I', 'S', 'Z', 'B']:
                    corrected[i] = self.char_map.get(corrected[i], corrected[i])
        
        # Positions 5-7 (series): should be letters - CRITICAL: Do NOT remap E, J, G here!
        # These are valid series letters, not digits
        # Only remap if it's clearly a digit in a letter position
        if len(corrected) >= 6:
            series_start = 4
            series_end = min(7, len(corrected) - 4)  # Leave last 4 for number
            for i in range(series_start, series_end):
                if corrected[i].isdigit():
                    # In series position, try reverse mapping (6->G, 3->J, 9->E)
                    if corrected[i] in ['6', '3', '9']:
                        corrected[i] = self.reverse_char_map.get(corrected[i], corrected[i])
        
        # Last 4 positions: should be digits (number)
        if len(corrected) >= 8:
            for i in range(len(corrected)-4, len(corrected)):
                if corrected[i].isalpha():
                    # Only remap commonly confused letters in number position
                    if corrected[i] in ['O', 'I', 'S', 'Z', 'B', 'G', 'J', 'T']:
                        corrected[i] = self.char_map.get(corrected[i], corrected[i])
        
        corrected_text = ''.join(corrected)
        return self._parse_direct(corrected_text)
    
    def _parse_rto_fixed(self, text: str) -> Optional[str]:
        """Fix common RTO OCR errors - but preserve E, J, G in series positions"""
        if len(text) < 7:
            return None
        
        state = text[:2]
        rto_raw = text[2:4]
        rest = text[4:]  # This includes series + number
        
        # Apply RTO fixes (only for RTO position, not series)
        rto = self.rto_fixes.get(rto_raw, rto_raw)
        
        # Ensure RTO is digits (positions 2-3)
        if not rto.isdigit():
            # Only remap commonly confused chars in RTO position
            rto = ''.join(self.char_map.get(c, c) if c.isalpha() and c in ['O', 'I', 'S', 'Z', 'B'] else c for c in rto)
        
        # Ensure state is letters (positions 0-1)
        state_fixed = ''.join(self.reverse_char_map.get(c, c) if c.isdigit() else c for c in state)
        
        if state_fixed in self.valid_states and rto.isdigit() and len(rto) == 2:
            reconstructed = state_fixed + rto + rest
            return self._parse_direct(reconstructed)
        
        return None
    
    def _parse_remapped(self, text: str) -> Optional[str]:
        """Comprehensive character remapping - but preserve E, J, G in series positions"""
        # CRITICAL: Only remap in specific positions, not in series positions
        if len(text) < 6:
            return None
        
        remapped = list(text)
        
        # Positions 0-1 (state): digits -> letters
        for i in range(min(2, len(remapped))):
            if remapped[i].isdigit():
                remapped[i] = self.reverse_char_map.get(remapped[i], remapped[i])
        
        # Positions 2-3 (RTO): letters -> digits (only commonly confused)
        for i in range(2, min(4, len(remapped))):
            if remapped[i].isalpha() and remapped[i] in ['O', 'I', 'S', 'Z', 'B']:
                remapped[i] = self.char_map.get(remapped[i], remapped[i])
        
        # Positions 4+ (series + number): be careful
        # Series positions (4-6): digits -> letters (6->G, 3->J, 9->E)
        if len(remapped) >= 6:
            for i in range(4, min(7, len(remapped) - 4)):  # Series area
                if remapped[i].isdigit() and remapped[i] in ['6', '3', '9']:
                    remapped[i] = self.reverse_char_map.get(remapped[i], remapped[i])
        
        # Last 4 positions (number): letters -> digits
        if len(remapped) >= 8:
            for i in range(len(remapped)-4, len(remapped)):
                if remapped[i].isalpha() and remapped[i] in self.char_map:
                    remapped[i] = self.char_map.get(remapped[i], remapped[i])
        
        return self._parse_direct(''.join(remapped))
    
    def _reconstruct_from_pattern(self, text: str) -> Optional[str]:
        """
        Reconstruct plate using regex pattern matching
        Based on plate_ocr_paddle.py reconstruction logic
        Prioritizes 2-letter series (most common format)
        """
        # Try direct regex match - prioritize 2-letter series first
        match = re.search(self.plate_regex, text)  # 2-letter series
        if match:
            state, rto, series, number = match.groups()
            if self._validate_parts(state, rto, series, number, raw_text_length=len(text)):
                return self._format_plate(state, rto, series, number)
        
        # Try flexible regex for 1-letter series (fallback)
        match = re.search(self.plate_regex_flexible, text)
        if match:
            state, rto, series, number = match.groups()
            if self._validate_parts(state, rto, series, number, raw_text_length=len(text)):
                return self._format_plate(state, rto, series, number)
        
        # Try with character removal for cases like TS09ET9099 (should be TS 09 ET 9099)
        # or BRO1C7F8627 (should be BR 01 CF 8627 - remove the 7)
        if len(text) >= 10:
            # Try removing single characters/digits that might be OCR errors
            for remove_pos in range(4, len(text) - 4):
                test_text = text[:remove_pos] + text[remove_pos+1:]
                # Try 2-letter series first
                match = re.search(self.plate_regex, test_text)
                if match:
                    state, rto, series, number = match.groups()
                    if self._validate_parts(state, rto, series, number, raw_text_length=len(text)):
                        return self._format_plate(state, rto, series, number)
                # Try flexible
                match = re.search(self.plate_regex_flexible, test_text)
                if match:
                    state, rto, series, number = match.groups()
                    if self._validate_parts(state, rto, series, number, raw_text_length=len(text)):
                        return self._format_plate(state, rto, series, number)
        
        # Partial recovery (suffix) - from plate_ocr_paddle.py
        # Try to extract series + number from end (can be 3-6 chars: 1-2 letter series + 1-4 digit number)
        if len(text) >= 6:
            # Try 2-letter series + 1-4 digit number (3-6 chars total)
            for num_len in [4, 3, 2, 1]:  # Try 4 digits first, then 3, 2, 1
                if len(text) >= 4 + num_len:
                    suffix = text[-(2 + num_len):]  # Last 2+num_len chars
                    if re.match(r"[A-Z]{2}[0-9]{" + str(num_len) + r"}", suffix):
                        series = suffix[:2]
                        number = suffix[2:]
                        # Try to extract state and RTO from beginning
                        if len(text) >= 4 + 2 + num_len and text[:2].isalpha():
                            state = text[:2]
                            if state in self.valid_states:
                                rto_part = text[2:4]
                                if rto_part.isdigit() and len(rto_part) == 2:
                                    if self._validate_parts(state, rto_part, series, number, raw_text_length=len(text)):
                                        return self._format_plate(state, rto_part, series, number)
        
        return None
    
    def _validate_parts(self, state: str, rto: str, series: str, number: str, raw_text_length: int = 0) -> bool:
        """Validate plate parts - STRICT validation for Indian plate format AA BB CC DDDD"""
        # Part 1 (AA): State code - Must be exactly 2 letters and valid
        if len(state) != 2 or not state.isalpha() or state not in self.valid_states:
            return False
        
        # Part 2 (BB): RTO code - Must be exactly 2 digits
        if len(rto) != 2 or not rto.isdigit():
            return False
        
        # Part 3 (CC): Series - REQUIRED when raw text length >= 8
        # CRITICAL: Series is REQUIRED for valid Indian plates when text suggests it
        if not series or len(series) == 0:
            # Only allow no series if raw text is very short (6-7 chars) suggesting old format
            # If raw_text_length >= 8, series MUST exist (e.g., "KL 30 6392" is invalid, should be "KL 30 G 6392")
            if raw_text_length >= 8:
                return False  # Reject plates missing series when text suggests it should exist
        else:
            # Series must be 1-3 letters, no O or I
            if len(series) > 3 or not series.isalpha() or any(c in 'OI' for c in series):
                return False
        
        # Part 4 (DDDD): Number - Must be 1-4 digits (can be 1, 2, 3, or 4 digits)
        # CRITICAL: Indian plates can have 1-4 digit numbers (1-9999)
        if not number.isdigit() or len(number) < 1 or len(number) > 4:
            return False
        
        # If 4 digits, cannot start with 0
        if len(number) == 4 and number[0] == '0':
            return False
        
        # CRITICAL: If raw text is 9+ chars, series MUST exist
        # This prevents accepting "KL 30 6392" (missing series) when it should be "KL 30 G 6392"
        if raw_text_length >= 9 and not series:
            return False
        
        return True
    
    def _format_plate(self, state: str, rto: str, series: str, number: str) -> str:
        """Format validated parts"""
        formatted = f"{state} {rto}"
        if series:
            formatted += f" {series}"
        formatted += f" {number.zfill(4)}"
        return formatted
    
    def _extract_parts(self, formatted: str) -> Dict:
        """Extract parts from formatted plate"""
        parts_list = formatted.split()
        return {
            'state': parts_list[0] if len(parts_list) > 0 else '',
            'rto': parts_list[1] if len(parts_list) > 1 else '',
            'series': parts_list[2] if len(parts_list) == 4 else '',
            'number': parts_list[-1] if len(parts_list) >= 3 else ''
        }
    
    def _calculate_quality_score(self, parts: Dict, confidence: float) -> float:
        """Calculate quality score (0-15)"""
        score = 0
        
        if parts['state'] in self.valid_states:
            score += 4
        if parts['rto'].isdigit() and len(parts['rto']) == 2:
            score += 3
        if parts['series']:
            score += 2
        # Part 4: Number can be 1-4 digits (1-9999)
        if parts['number'].isdigit() and 1 <= len(parts['number']) <= 4:
            score += 3
        score += min(3, confidence * 3)
        
        return score
    
    def _try_format_raw_text(self, raw_text: str) -> Optional[str]:
        """Try to format raw text that failed normalization but has valid state code"""
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        if len(cleaned) < 6:
            return None
        
        # Try to extract parts from raw text - prioritize 2-letter series
        state = cleaned[:2]
        if state not in self.valid_states:
            return None
        
        # Try different parsing strategies - prioritize 2-letter series
        for rto_start in range(2, min(5, len(cleaned))):
            rto = cleaned[rto_start:rto_start+2]
            if rto.isdigit():
                remaining = cleaned[rto_start+2:]
                if len(remaining) >= 4:
                    # Try with 2-letter series first (most common)
                    if len(remaining) >= 6:
                        series = remaining[:2]
                        number = remaining[2:6]
                        if number.isdigit() and series.isalpha() and len(number) == 4:
                            if self._validate_parts(state, rto, series, number, raw_text_length=len(cleaned)):
                                return self._format_plate(state, rto, series, number)
                    # Try with 1-letter series
                    if len(remaining) >= 5:
                        series = remaining[:1]
                        number = remaining[1:5]
                        if number.isdigit() and series.isalpha() and len(number) == 4:
                            if self._validate_parts(state, rto, series, number, raw_text_length=len(cleaned)):
                                return self._format_plate(state, rto, series, number)
                    # Try with no series
                    number = remaining[-4:]
                    if number.isdigit() and len(number) == 4:
                        if self._validate_parts(state, rto, '', number, raw_text_length=len(cleaned)):
                            return self._format_plate(state, rto, '', number)
        
        return None


# Wrapper for compatibility
class PlateOCR:
    def __init__(self, languages=['en'], use_gpu=False):
        self.ocr = UltimateOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
        self.languages = languages
        self.use_gpu = use_gpu
    
    def read_plate(self, plate_image, verbose=False):
        return self.ocr.read_plate(plate_image, verbose=verbose)