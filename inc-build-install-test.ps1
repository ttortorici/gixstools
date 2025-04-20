# Step 1: Increment the version number in pyproject.toml
Write-Host "Updating version number in pyproject.toml..."
$currentVersion = (Get-Content pyproject.toml | Select-String -Pattern '^version = "(.*)"' | ForEach-Object { $_.Matches.Groups[1].Value })
if (-not $currentVersion) {
    Write-Error "Failed to find the current version in pyproject.toml."
    exit 1
}

# Split the version into major, minor, and patch
$versionParts = $currentVersion -split '\.'
$major = [int]$versionParts[0]
$minor = [int]$versionParts[1]
$patch = [int]$versionParts[2]
$patch += 1
$newVersion = "$major.$minor.$patch"

# Update the version in pyproject.toml
(Get-Content pyproject.toml) -replace "version = `"$currentVersion`"", "version = `"$newVersion`"" | Set-Content pyproject.toml
Write-Host "Updated version to $newVersion."

# Step 2: Build the Python package
Write-Host "Building the Python package..."
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build the package."
    exit 1
}

# Step 3: Find the latest .whl file in the dist directory
Write-Host "Finding the latest .whl file..."
$latestWhl = Get-ChildItem -Path dist\*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latestWhl) {
    Write-Error "No .whl file found in the dist directory."
    exit 1
}
Write-Host "Found latest .whl file: $($latestWhl.Name)"

# Step 4: Install the latest .whl file using pip
Write-Host "Installing the latest .whl file..."
pip install $latestWhl.FullName
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install the package."
    exit 1
}

# Step 5: Run the tests
Write-Host "Running tests..."
pip install -U nose2
python -m nose2
if ($LASTEXITCODE -ne 0) {
    Write-Error "Tests failed."
    exit 1
}

Write-Host "Build, installation, and tests completed successfully."