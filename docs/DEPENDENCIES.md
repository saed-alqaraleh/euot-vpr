# Dependency scope

**Statistics-only CPU quick start:** Python 3.11/3.12 + NumPy 2.x. During package preparation, NumPy 2.3.5 was used for the statistical reproduction. `requirements-cpu.txt` is a compatibility range, **not a historical executed-run lock**.

**CPU model-equivalence tests:** add PyTorch CPU (tested PyTorch 2.10.0+cpu), Pillow (tested 12.3.0); tests rely on PyTorch serialized `.pt` files and load author-fitted head state dictionaries with `weights_only=True`.

**Optional plots:** Matplotlib (tested 3.10.8). Two figures call the controlled generator and require the excluded third-party photographic image bank; other numerical figures use archived JSON/hit evidence.

**Recorded controlled original:** Python 3.11.2; NumPy 2.1.3; OpenCV 4.11.0; Matplotlib 3.10.1; PyTorch not available; Linux CPU (from supplied environment record, not included because it contains staging metadata). This version set concerns the controlled benchmark, **not MSLS-val GPU heads**.

**Recorded MSLS-val v4:** Python >=3.10; PyTorch >=2.1; torchvision; NumPy; Pillow; scikit-learn; released SALAD DINOv2 trunk. SALAD upstream indicates a tested PyTorch 2.1.0 / CUDA 12.1 setup; dependencies include its own environment and `pytorch-lightning`, `pytorch-metric-learning`. Use upstream installations from the official repository and authorized checkpoint source. No full executed v6 GPU environment lock is in the supplied scientific evidence, so one is not invented.
