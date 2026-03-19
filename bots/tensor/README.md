`tensor` is the training workspace, not the deploy target.

It contains training code, sessions, logs, and other artifacts that should not be uploaded with `cambc test-run` or `cambc submit`.

Use `bots/tensor_inference` for:

- `cambc run`
- `cambc test-run`
- `cambc submit`

If you train a new model, export the chosen `weights.pkl` into `bots/tensor_inference/weights.json` before testing or submitting.
