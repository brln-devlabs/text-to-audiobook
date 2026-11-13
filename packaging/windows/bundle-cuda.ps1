# Collect the CUDA build and the CUDA DLLs it needs into one folder, which
# the CUDA packages install as {app}\cuda (the normal build looks for it
# there, src/gpu.rs).
#
#   packaging\windows\bundle-cuda.ps1 -Binary target\release\text-to-audiobook.exe `
#       -CudaBin $env:CUDA_PATH\bin -Out dist\windows\cuda
#
# Copied are all DLLs from the CUDA bin folder that the binary needs,
# directly or through another CUDA DLL. The NVIDIA driver (nvcuda.dll)
# comes with the driver and the system DLLs with Windows.
param(
    [Parameter(Mandatory)][string]$Binary,
    [Parameter(Mandatory)][string]$CudaBin,
    [Parameter(Mandatory)][string]$Out
)
$ErrorActionPreference = 'Stop'

$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
$dumpbin = (Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" | Select-Object -First 1).FullName

function Get-Dependencies([string]$File) {
    & $dumpbin /nologo /dependents $File |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^\S+\.dll$' }
}

$deps = @(Get-Dependencies $Binary)
Write-Host "Dependencies of ${Binary}:"
$deps | ForEach-Object { Write-Host "  $_" }
if (-not ($deps -match '^nvcuda\.dll$')) {
    throw "$Binary does not use nvcuda.dll: not a CUDA build?"
}
if ($deps -match '^(vcruntime|msvcp|vcomp)') {
    throw "$Binary needs the Visual C++ runtime (static C runtime expected)"
}

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
Copy-Item $Binary (Join-Path $Out 'text-to-audiobook-cuda.exe')

$copied = @{}
$queue = $deps
while ($queue.Count -gt 0) {
    $next = @()
    foreach ($dll in $queue) {
        $key = $dll.ToLowerInvariant()
        $src = Join-Path $CudaBin $dll
        # System DLLs and nvcuda.dll are not in the CUDA bin folder.
        if ($copied.ContainsKey($key) -or -not (Test-Path $src)) { continue }
        Copy-Item $src $Out
        $copied[$key] = $true
        $next += @(Get-Dependencies $src)
    }
    $queue = $next
}

# NVRTC loads its builtins DLL at run time, not through the import table.
if (@($copied.Keys) -match '^nvrtc64_') {
    Get-ChildItem $CudaBin -Filter 'nvrtc-builtins64_*.dll' |
        Where-Object { $_.Name -notmatch '\.alt\.' } |
        Copy-Item -Destination $Out
}
# cuBLAS (and cuBLASLt through it) is always needed. The CUDA runtime and
# NVRTC only if something uses them: unlike the Linux linker, MSVC leaves out
# import libraries nobody calls (2026-11: nvcuda, cublas64_12, curand64_10).
foreach ($must in '^cublas64_', '^cublaslt64_') {
    if (-not (@($copied.Keys) -match $must)) {
        throw "no DLL matching $must copied from ${CudaBin}"
    }
}

Get-ChildItem $Out | Format-Table Name, @{ n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } }
