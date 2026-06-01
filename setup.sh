#!/usr/bin/env bash
# One-shot setup: creates a project-local .venv and installs runtime dependencies.
# Training happens in the Colab notebook (notebooks/01_train_vgg16.ipynb).
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip ..."
python -m pip install --upgrade pip --quiet

echo "Installing requirements ..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate with:  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Train the model in Colab → notebooks/01_train_vgg16.ipynb"
echo "  2. Download vgg16_model.onnx + metrics.json + figures and drop into models/ + reports/"
echo "  3. Run the demo:  streamlit run app/streamlit_app.py"
