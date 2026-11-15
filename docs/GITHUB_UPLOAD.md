# Import and verify the repository on GitHub

An empty private repository must be created in advance. On a computer with Git and permission to push to that repository, download the companion Git bundle and run:

```bash
git clone -b main EUOT-VPR_public_candidate.git.bundle euot-vpr
cd euot-vpr
git remote remove origin
git remote add origin https://github.com/saed-alqaraleh/euot-vpr.git
git push -u origin main
```

Leave the repository **private** until all file-level and rights checks have been completed. After pushing, confirm that the `main` branch includes the complete directory tree, that the GitHub Actions verification job passes, and that no image bank, manuscript, access token or internal note was included.

For an offline integrity check before the push:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/verify_release.py
python scripts/reproduce_headline.py --output outputs/headline_msls.json
```

The original source-image datasets and third-party pretrained SALAD checkpoint are not part of the release. Consult `docs/THIRD_PARTY_RIGHTS.md` and `docs/IMAGE_LEVEL_REPLAY.md` before public visibility.
