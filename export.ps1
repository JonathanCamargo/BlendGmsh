# Export blendgmsh as a .zip for Blender extension install (Windows)
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot

# Clear stale __pycache__ from installed extension
$ExtDir = Join-Path $env:APPDATA "Blender Foundation\Blender\5.1\extensions\user_default\blendgmsh"
if (Test-Path $ExtDir) {
    Get-ChildItem $ExtDir -Directory -Recurse -Filter __pycache__ |
        Remove-Item -Recurse -Force
}

$TmpDir  = Join-Path $env:TEMP "blendgmsh"
$ZipPath = Join-Path $env:TEMP "blendgmsh.zip"

if (Test-Path $TmpDir)  { Remove-Item $TmpDir  -Recurse -Force }
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

Copy-Item -Recurse blender_addon $TmpDir
Copy-Item -Recurse step_converter (Join-Path $TmpDir "step_converter")
Copy-Item -Recurse matching_library (Join-Path $TmpDir "matching_library")
Copy-Item -Recurse schema (Join-Path $TmpDir "schema")

Get-ChildItem $TmpDir -Directory -Recurse -Filter __pycache__ |
    Remove-Item -Recurse -Force

Compress-Archive -Path $TmpDir -DestinationPath $ZipPath
Remove-Item $TmpDir -Recurse -Force

Pop-Location
Write-Host $ZipPath
