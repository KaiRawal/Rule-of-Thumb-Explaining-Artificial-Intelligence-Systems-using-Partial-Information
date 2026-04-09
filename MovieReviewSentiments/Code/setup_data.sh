# Setup script to download and extract ERASER movies dataset


MOVIES_DIR="./DATA/movies"
DOWNLOAD_URL="https://www.eraserbenchmark.com/zipped/movies.tar.gz"
TAR_FILE="movies.tar.gz"

rm -rf "$MOVIES_DIR"

echo "Setting up ERASER movies dataset..."

# Create DATA directory if it doesn't exist
mkdir -p ./DATA

# # Check if movies directory already exists with required files
# if [ -d "$MOVIES_DIR" ] && [ -f "$MOVIES_DIR/train.jsonl" ]; then
#     echo "✓ Movies dataset already exists at $MOVIES_DIR"
#     exit 0
# fi

# Download the dataset
echo "Downloading movies dataset from $DOWNLOAD_URL..."
if command -v curl &> /dev/null; then
    curl -L -o "$TAR_FILE" "$DOWNLOAD_URL"
elif command -v wget &> /dev/null; then
    wget -O "$TAR_FILE" "$DOWNLOAD_URL"
else
    echo "Error: Neither curl nor wget found. Please install one to download the dataset."
    exit 1
fi

# Extract to DATA directory
echo "Extracting dataset to $MOVIES_DIR..."
tar -xzf "$TAR_FILE" -C ./DATA

# Clean up
rm "$TAR_FILE"

echo "✓ Dataset setup complete!"
