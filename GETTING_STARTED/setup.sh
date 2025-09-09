#!/bin/bash
# RadarSim Getting Started Setup Script
# Sets up the environment for running RadarSim examples

echo "========================================="
echo "RadarSim Getting Started Setup"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "run_first_simulation.py" ]; then
    echo "Error: Please run this script from the GETTING_STARTED directory"
    exit 1
fi

# Create a local virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
cat > requirements.txt << EOF
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pyyaml>=5.4.0
dataclasses
EOF

pip install -r requirements.txt

# Check if maturin is needed (for Rust extensions)
if [ -f "../Cargo.toml" ]; then
    echo "Installing maturin for Rust extensions..."
    pip install maturin
    
    # Build Rust extensions if present
    if [ -d "../rust" ]; then
        echo "Building Rust extensions..."
        cd ..
        maturin develop --release
        cd GETTING_STARTED
    fi
fi

# Create a simple test to verify installation
echo "Testing installation..."
cd ..  # Go to project root
python3 -c "
import sys
import os
try:
    from src.radar import Radar, RadarParameters
    from src.target import Target
    from src.environment import Environment
    print('✓ Core modules loaded successfully')
except ImportError as e:
    print(f'✗ Failed to load modules: {e}')
    sys.exit(1)
"
TEST_RESULT=$?
cd GETTING_STARTED  # Return to GETTING_STARTED

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "To run your first simulation:"
    echo "  python run_first_simulation.py"
    echo ""
    echo "To run a YAML scenario:"
    echo "  python ../run_scenario.py first_scenario.yaml"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi