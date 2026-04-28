# Check if the venv directory exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists."
}

# Activate the virtual environment and install dependencies
Write-Host "Activating virtual environment and installing dependencies..."
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Write-Host "Setup complete. You can now run the project."
