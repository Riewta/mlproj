from matplotlib import font_manager, pyplot as plt

THAI_FONTS = ["Thonburi", "Tahoma", "Loma", "Garuda", "Sarabun", "Noto Sans Thai"]


def use_thai_font():
    """Matplotlib's default font has no Thai glyphs. Pick the first installed Thai font.

    On Colab: !apt-get -qq install fonts-thai-tlwg, then restart the runtime.
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    name = next((f for f in THAI_FONTS if f in installed), None)
    if name:
        plt.rcParams["font.family"] = name
    return name
