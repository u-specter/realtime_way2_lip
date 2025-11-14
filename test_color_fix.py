#!/usr/bin/env python3
"""
Test script to diagnose color artifacts in Wav2Lip output
"""

import cv2
import numpy as np
import openvino as ov
import os

# Load OpenVINO model
print("Loading OpenVINO model...")
core = ov.Core()
model_path = "./openvino_model/wav2lip_openvino_model.xml"
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

# Create dummy inputs
print("Creating test inputs...")
mel_batch = np.random.randn(1, 1, 80, 16).astype(np.float32)
img_batch = np.random.randn(1, 6, 96, 96).astype(np.float32)

# Run inference
print("Running inference...")
output = compiled_model([mel_batch, img_batch])['output']

print(f"\n=== MODEL OUTPUT ANALYSIS ===")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
print(f"Output mean: {output.mean():.6f}")

# Convert to image format
output_img = output[0].transpose(1, 2, 0) * 255.0
output_img = output_img.astype(np.uint8)

print(f"\nAfter scaling to 0-255:")
print(f"Image shape: {output_img.shape}")
print(f"Image dtype: {output_img.dtype}")
print(f"Image range: [{output_img.min()}, {output_img.max()}]")

# Sample pixel in center
center_y, center_x = 48, 48
pixel_rgb = output_img[center_y, center_x, :]
print(f"\nCenter pixel (RGB format from model): {pixel_rgb}")

# Test RGB2BGR conversion
img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
pixel_bgr = img_bgr[center_y, center_x, :]
print(f"After RGB2BGR: {pixel_bgr}")

# Create a test mouth region image
print("\n=== CREATING TEST IMAGE ===")
test_mouth = np.ones((96, 96, 3), dtype=np.uint8) * 200  # Light gray
test_mouth[30:70, 30:70, :] = [150, 120, 100]  # Skin-like color in center

# Save test images
os.makedirs("debug_output", exist_ok=True)
cv2.imwrite("debug_output/01_model_output_rgb.jpg", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
cv2.imwrite("debug_output/02_test_mouth.jpg", test_mouth)

# Test blending
alpha = 0.85
blended = cv2.addWeighted(img_bgr, alpha, test_mouth, 1 - alpha, 0)
cv2.imwrite("debug_output/03_blended.jpg", blended)

print(f"\nTest images saved to debug_output/")
print(f"- 01_model_output_rgb.jpg (model output)")
print(f"- 02_test_mouth.jpg (test skin color)")
print(f"- 03_blended.jpg (blended result)")

print("\n=== DIAGNOSIS ===")
if output.max() > 1.0:
    print("WARNING: Model output > 1.0 (expected 0-1 from Sigmoid)")
if output.min() < 0.0:
    print("WARNING: Model output < 0.0 (expected 0-1 from Sigmoid)")

# Check if OpenVINO conversion preserved color channels correctly
print(f"\nChannel statistics:")
for i, name in enumerate(['R', 'G', 'B']):
    print(f"  {name}: mean={output_img[:,:,i].mean():.1f}, std={output_img[:,:,i].std():.1f}")
