param(
    [string]$FijiDir = "C:\Program Files\Fiji",
    [string]$PythonExecutable = "python",
    [string]$JdkHome = "",
    [string]$OutputJar = "",
    [string]$WorkDir = "",
    [switch]$RunSmoke
)

# Compatibility wrapper; the cross-platform builder always runs the smoke suite.
# No installed Fiji files or unrelated plugin JARs are changed.
$ErrorActionPreference = "Stop"
$moduleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $moduleRoot)
$builder = Join-Path $repositoryRoot "scripts/build_fiji.py"
$arguments = @($builder, "--fiji", $FijiDir)
if ($JdkHome) { $arguments += @("--jdk", $JdkHome) }
if ($OutputJar) { $arguments += @("--output", $OutputJar) }
if ($WorkDir) { $arguments += @("--work-dir", $WorkDir) }
& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "Fiji build/smoke failed with exit code $LASTEXITCODE" }
