`tensor_inference` is the production deploy bot.

Use this folder for local runs, remote test runs, and submissions:

```bash
.venv/bin/cambc run tensor_inference starter
.venv/bin/cambc test-run tensor_inference starter
.venv/bin/cambc submit bots/tensor_inference
```

Folder contents:

- `main.py`: inference-only bot entrypoint
- `weights.json`: bundled deploy weights
- `tensor_lib/`: stdlib-only inference library

To refresh the deployed weights from a training session:

```bash
python3.13 -c "
import json, pickle
w = pickle.load(open('bots/tensor/sessions/<session>/weights.pkl', 'rb'))
json.dump(w, open('bots/tensor_inference/weights.json', 'w'))
"
```

Do not upload `bots/tensor` directly. That folder is the training workspace and contains non-deploy artifacts.
