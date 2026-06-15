# FaceMorph 🧑↔️👩

**Morph any face along the female ↔ male axis with a slider — live, on your own machine.**

FaceMorph is a small desktop app around a **StarGAN** trained on the CelebA face dataset. Load a
photo (or pick a bundled one), drag the **Female ↔ Male** slider, and the network re-renders the
face at the chosen gender while keeping it the **same person**.

> Made for an Advanced Python course at Kozminski University. Runs anywhere — **no GPU, no
> TensorFlow** — thanks to ONNX Runtime.

---

## ✨ Features
- Live **Female ↔ Male** slider morphing
- **Surprise me** — morph bundled sample faces
- **Load photo** — use your own picture
- **Save** the morph, **export** a 7-step strip, or an animated **GIF**

## 🚀 Run it

### ⭐ Windows — one click, no Python needed
1. Download **[`FaceMorph.exe`](FaceMorph.exe)** (~76 MB): click the file above, then the **Download** button.
2. Double-click it. Done — the app opens.

> The model is baked into the `.exe`, so there is **nothing to install**: no Python, no `pip`.
> First launch unpacks once and takes a few seconds. If Windows shows a blue **SmartScreen**
> box ("Windows protected your PC"), click **More info → Run anyway** — it is our own unsigned
> app, not a virus.

### 🍎 macOS — one click
1. Open the [**Releases**](../../releases) page and download the file for your Mac:
   - **Apple Silicon** (M1 / M2 / M3): `FaceMorph-macOS-AppleSilicon.zip`
   - **Intel** Mac: `FaceMorph-macOS-Intel.zip`
2. Unzip, then **right-click `FaceMorph.app` → Open** the first time (the app is unsigned, so a plain
   double-click is blocked by Gatekeeper). If macOS calls it "damaged", run once in Terminal:
   `xattr -cr /path/to/FaceMorph.app`

> The Mac and Windows apps are built automatically in the cloud (GitHub Actions) and published on the
> Releases page, so they track the code. No Mac or build tools needed on your side.

### 🛠 Run from the source (Linux, or to tinker)
You need **Python 3.9+**:
```bash
pip install -r requirements.txt
python FaceMorph.py
```
macOS: double-click `Uruchom_macOS.command` (first time: right-click → Open).

A morph takes well under a second on a normal laptop CPU.

## 📦 What's inside
| File | Purpose |
|---|---|
| `FaceMorph.exe` | **one-click Windows app** — Python, libraries and the model all bundled |
| `FaceMorph.py` | the GUI (source) |
| `morph_onnx.py` | inference engine (ONNX Runtime) |
| `generator.onnx` | the trained model |
| `discriminator.onnx` | optional — auto-detects gender to set the slider start |
| `assets/samples/` | bundled aligned faces for "Surprise me" |
| `requirements.txt` | `onnxruntime`, `customtkinter`, `Pillow`, `numpy` |

## 🧠 How it works (short version)
A StarGAN generator takes the image plus a target-gender value (as an extra channel) and outputs
the morph. **Cycle** and **identity** losses keep the person recognizable; the gender label is
trained on continuous values, so the slider gives smooth in-between faces. The model is 128×128,
trained on the full CelebA (~202k images). Quantitatively: ~99% gender-swap success and low
identity/cycle error on the validation set.

## ⚠️ Notes & honest limitations
- 128×128 resolution — recognizable and smooth, not photorealistic up close.
- Works best on **front-facing, centred** faces; unusual poses/lighting may look off.
- "Load photo" uses a simple centre-crop in this portable build; the bundled **Surprise me** faces
  are pre-aligned and look best.

## 📄 Data, license & ethics
- The model is trained on **CelebA** (Liu et al., ICCV 2015), which is for **non-commercial
  research** — please use this model and app accordingly.
- This is an educational research demo, not an identity-forgery / deepfake tool.
- Code: free to use for learning. Faces in `assets/samples/` are CelebA crops.

*Built with help from an AI coding assistant for the engineering/packaging; the model design and
machine-learning choices are the team's.*
