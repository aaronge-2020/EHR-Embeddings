"""
Setup script for EHR Embeddings project
"""
import subprocess
import sys
import os
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "embeddings_cache", 
        "models",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_env_file():
    """Create .env file from template"""
    env_content = """# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Model Configuration
EMBEDDING_MODEL=models/embedding-001
BATCH_SIZE=100
MAX_RETRIES=3

# Data Configuration
EHR_DATA_PATH=data/ehr_data.csv
EMBEDDINGS_CACHE_DIR=embeddings_cache/
MODEL_OUTPUT_DIR=models/

# Logging
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("⚠️  Please edit .env file and add your Google API key")
    else:
        print("ℹ️  .env file already exists")

def setup_jupyter():
    """Set up Jupyter kernel"""
    try:
        subprocess.check_call([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "ehr_embeddings", "--display-name", "EHR Embeddings"])
        print("✅ Jupyter kernel installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Could not install Jupyter kernel: {e}")

def main():
    """Main setup function"""
    print("EHR Embeddings Project Setup")
    print("=" * 40)
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing Python packages...")
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Create environment file
    print("\n3. Setting up environment configuration...")
    create_env_file()
    
    # Setup Jupyter
    print("\n4. Setting up Jupyter environment...")
    setup_jupyter()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your Google API key")
    print("2. Run 'python example_usage.py' to test the setup")
    print("3. Launch Jupyter: 'jupyter notebook notebook_example.ipynb'")
    print("\nAPI Key Instructions:")
    print("- Get your Google API key from: https://makersuite.google.com/app/apikey")
    print("- Add it to the .env file: GOOGLE_API_KEY=your_actual_key_here")

if __name__ == "__main__":
    main() 