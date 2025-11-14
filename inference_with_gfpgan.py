"""
OPTIONAL ENHANCEMENT: Wav2Lip + GFPGAN для улучшения качества
Использовать ТОЛЬКО если качество текущего решения недостаточно

Установка:
pip3 install gfpgan facexlib realesrgan
wget -P checkpoints https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
"""

# Добавить в класс Wav2LipInference после строки 95:

def load_gfpgan_model(self):
    """
    OPTIONAL: Load GFPGAN for face enhancement
    Улучшает качество лица после Wav2Lip генерации
    """
    try:
        from gfpgan import GFPGANer

        gfpgan = GFPGANer(
            model_path='checkpoints/GFPGANv1.4.pth',
            upscale=1,  # No upscaling, just enhancement
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )
        print("[GFPGAN] Face enhancement model loaded")
        return gfpgan
    except Exception as e:
        print(f"[GFPGAN] Not available: {e}")
        return None


# Добавить в __init__ после строки 95:
# self.gfpgan = self.load_gfpgan_model()


# ЗАМЕНИТЬ блок обработки frames (строки 367-383) на:

"""
for idx, (p, f, c) in enumerate(zip(pred, frames, coords)):
    y1, y2, x1, x2 = c

    # Model already outputs BGR format (OpenVINO conversion)
    p = p.astype(np.uint8)
    p = cv2.resize(p, (x2 - x1, y2 - y1))

    # OPTIONAL: Enhance with GFPGAN (if available)
    if hasattr(self, 'gfpgan') and self.gfpgan is not None:
        try:
            # GFPGAN enhancement
            _, _, p_enhanced = self.gfpgan.enhance(
                p,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )
            if p_enhanced is not None:
                p = p_enhanced
                if idx == 0:
                    print("[GFPGAN] Face enhanced successfully")
        except Exception as e:
            if idx == 0:
                print(f"[GFPGAN] Enhancement failed: {e}")
            pass

    # Direct replacement (official OpenVINO method)
    f[y1:y2, x1:x2] = p

    generated_frames.append(f.copy())
"""

# PERFORMANCE IMPACT:
# - GFPGAN adds ~50-100ms per frame
# - Total latency: 300ms + 50ms = 350ms (still acceptable)
# - Quality improvement: significant (removes artifacts, sharpens face)

# КОГДА ИСПОЛЬЗОВАТЬ:
# 1. Если видны артефакты на лице после Wav2Lip
# 2. Если лицо выглядит размытым
# 3. Для банковской презентации (professional quality)

# КОГДА НЕ ИСПОЛЬЗОВАТЬ:
# 1. Если текущее качество достаточное
# 2. Если latency критичен (< 300ms требование)
# 3. Если CPU слабый (MacBook Pro - OK, RPi - NO)
