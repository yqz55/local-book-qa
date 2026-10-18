$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m streamlit run app.py --server.address 127.0.0.1
