# Last-Mile Verifier - Payment Screenshot Tamper Detection (PyTorch)

## What is inside
| file | purpose |
|---|---|
| `generate_synthetic_data.py` | creates fake UPI-style receipts: genuine vs edited (retyped text, copy-move, blur patch) |
| `ela.py` | Error Level Analysis + preprocessing (RGB + ELA = 6-channel input) |
| `model.py` | `TamperNet` - small CNN, avg+max pooling (edits are small local regions) |
| `dataset.py`, `train.py` | training with accuracy / precision / recall / F1 / AUC on val + test |
| `predict.py` | single image -> P(tampered) + ELA heatmap |
| `app.py` | Flask web app: upload -> unique link -> merchant cross-check -> timestamped JSON record |

## Open in VS Code (quickest way)
1. Unzip, then **File > Open Folder...** and choose the `last_mile_verifier` folder.
2. Install the suggested **Python** extension when VS Code asks.
3. Windows: double-click `setup.bat` (Mac/Linux: `bash setup.sh`). It creates `venv` and installs everything.
4. `Ctrl+Shift+P` > **Python: Select Interpreter** > pick the one inside `venv`.
   (On Mac/Linux change the path in `.vscode/settings.json` to `venv/bin/python`.)
5. Open **Run and Debug** (Ctrl+Shift+D) and run in order:
   `1. Generate dataset` > `2. Train model` > `3. Predict one image` > `4. Run web app`
   (open http://127.0.0.1:5000 for the app).

## Run from terminal instead
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python generate_synthetic_data.py --n 3000 --out data
python train.py --data data --epochs 15              # GPU recommended, CPU works but is slow
python predict.py some_screenshot.jpg
python app.py                                        # open http://127.0.0.1:5000
```

## IMPORTANT - be honest in your demo / viva
* The model trained on synthetic data learns *my* fake templates. To work on real
  PhonePe / GPay / Paytm screenshots, **add real data** in the same folder layout
  (`data/train/genuine`, `data/train/tampered`, ...): collect real screenshots as genuine
  and create tampered versions yourself (edit with Photopea/Paint/Canva, change amount/txn ID).
  Even 300-500 real pairs added to the synthetic set help a lot. Mix, then fine-tune.
* No detector is 100% - the score is a *signal*. That's why the platform also cross-checks
  with the other party's confirmation; that cross-check is the strong evidence.
* Ideas for extra marks: OCR (`pytesseract`/`easyocr`) to auto-read amount + txn ID instead of typing,
  metadata check (EXIF software tag), Grad-CAM heatmap showing the edited region, PDF export of the record.
