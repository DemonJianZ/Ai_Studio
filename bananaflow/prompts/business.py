def build_business_prompt(mode: str, user_prompt: str, has_ref_image: bool) -> str:
    clean_prompt = (user_prompt or "").strip().strip(",").strip()
    style_suffix = f" Additional requirements: {clean_prompt}" if clean_prompt else ""

    if mode == "bg_replace":
        if has_ref_image:
            return f"""Change the background to match the reference image.
CRITICAL: Keep the main product and any hand holding it exactly as is.
Do not change the product or the hand gesture.
Only replace the background environment.{style_suffix}"""
        bg_desc = clean_prompt if clean_prompt else "clean professional studio background"
        return (
            f"Generate a background: '{bg_desc}'. "
            "CRITICAL: Isolate the foreground product and the hand holding it. "
            "Keep the product and hand pixels unchanged. Only replace the background."
        )

    if mode == "gesture_swap":
        if has_ref_image:
            return f"""Change the hand in the main image to match the reference image's hand.
CRITICAL: Keep the product object and the background exactly unchanged.
Only swap the hand and fingers to match the reference hand.{style_suffix}"""
        gesture_desc = clean_prompt if clean_prompt else "a natural hand holding gesture"
        return (
            f"Change the hand gesture to: '{gesture_desc}'. "
            "Constraint: Keep the product/object and the background exactly as is. Only change the hand."
        )

    if mode == "product_swap":
        if has_ref_image:
            return f"""Replace the object held in the hand with the product from the reference image.
CRITICAL: Keep the original hand gesture, skin tone, and background exactly unchanged.
Only swap the held object.{style_suffix}"""
        product_desc = clean_prompt if clean_prompt else "a generic product"
        return (
            f"Replace the held object with: '{product_desc}'. "
            "Constraint: Keep the hand gesture, skin tone, and background exactly as is."
        )

    if mode == "relight":
        return f"""
把整个手的颜色变成黄色调，不要有突兀的颜色不均匀现象，确保手和产品的光影自然融合在一起，像是在同一个环境下拍摄的一样。
保持手部的细节和质感，不要模糊或失真。确保手指的形状和位置与原图一致，不要改变手的姿势。
保持产品的外观和颜色不变，不要影响产品的细节。
确保背景和其他元素保持不变，不要引入新的物体或改变场景的构图。
{style_suffix}
"""

    if mode == "upscale":
        return f"""Upscale this image to high resolution and improve clarity.
Instruction: Denoise, sharpen details, and enhance textures.
{style_suffix}
CRITICAL: Maintain absolute fidelity to the original content.
Do not add new objects or change the subject's features.
The output must look like a high-end commercial photograph."""

    return f"Edit the image. {clean_prompt}"