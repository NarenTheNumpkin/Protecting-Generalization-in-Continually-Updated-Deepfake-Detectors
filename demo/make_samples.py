"""Copy the demo's sample faces from a GenD-preprocessed DeepFakeFace folder into samples/real and samples/fake.
The images are not redistributed with the repository; DeepFakeFace is available from its authors.
Usage: python make_samples.py [<DeepFakeFace_aligned folder>]  (default: $GEND_DATASETS/DeepFakeFace_aligned)"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import paths  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.DATASETS, "DeepFakeFace_aligned")
SAMPLES = {  # destination: (subset, unit)
    "real/real_01.png": ("real", "wiki_00_36672700_1987_12_10_2012"),
    "real/real_02.png": ("real", "wiki_01_17381201_1983_09_23_2011"),
    "real/real_03.png": ("real", "wiki_01_1765101_1911_10_05_1942"),
    "real/real_04.png": ("real", "wiki_01_27364101_1983_11_28_2012"),
    "fake/fake_text2img_01.png": ("text2img", "text2img_00_44126100_1934_12_17_2009"),
    "fake/fake_text2img_02.png": ("text2img", "text2img_02_742802_1913_04_12_2012"),
    "fake/fake_text2img_03.png": ("text2img", "text2img_02_36056702_1924_05_01_1984"),
    "fake/fake_inpainting_01.png": ("inpainting", "inpainting_02_19373302_1971_11_25_2014"),
    "fake/fake_inpainting_02.png": ("inpainting", "inpainting_02_13096702_1989_01_09_2014"),
    "fake/fake_faceswap_01.png": ("insight", "insight_25_22731525_1988_07_13_2014"),
    "fake/fake_faceswap_02.png": ("insight", "insight_00_2490600_1970_02_04_2013"),
}
for dst, (subset, unit) in SAMPLES.items():
    out = os.path.join(HERE, "samples", dst)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy(os.path.join(SRC, subset, unit, unit + ".png"), out)
print(f"copied {len(SAMPLES)} images to {os.path.join(HERE, 'samples')}")
