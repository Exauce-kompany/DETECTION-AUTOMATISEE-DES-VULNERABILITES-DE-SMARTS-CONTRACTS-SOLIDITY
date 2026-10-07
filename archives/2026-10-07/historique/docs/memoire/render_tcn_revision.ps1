param([string]$Stage = (Join-Path $PSScriptRoot 'tcn_staging'))

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$bundlePython = 'C:/Users/Exauce/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$rendererPath = 'C:/Users/Exauce/.codex/plugins/cache/openai-primary-runtime/documents/26.813.12317/skills/documents/render_docx.py'
$benchmarkRoot = Join-Path $workspaceRoot 'results/benchmark/cnn-xgboost-tcn-20261005'
$summary = Get-Content -LiteralPath (Join-Path $benchmarkRoot 'summary.json') -Raw | ConvertFrom-Json
if (-not $summary.complete) { throw 'The benchmark is incomplete. No thesis output was changed.' }
foreach ($modelName in @('cnn_bilstm', 'xgboost', 'tcn')) {
    $modelResult = $summary.models.$modelName
    if ($null -eq $modelResult.metrics -or $null -eq $modelResult.selected) {
        throw ('Missing saved metrics or checkpoint: ' + $modelName)
    }
}
if ($summary.paired_comparisons.pairs.Count -ne 3) { throw 'Three saved paired comparisons are required.' }

$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = 'C:/Program Files/LibreOffice/program;C:/Users/Exauce/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin;' + $env:PATH
function Invoke-ArtifactPython {
    param([string[]]$Arguments)
    & $bundlePython @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Artifact step failed: ' + $Arguments[0]) }
}

Invoke-ArtifactPython -Arguments @((Join-Path $PSScriptRoot 'update_tcn_models.py'), '--stage', $Stage)
Invoke-ArtifactPython -Arguments @((Join-Path $Stage 'make_figures.py'))
Invoke-ArtifactPython -Arguments @((Join-Path $Stage 'build_memoire.py'))
Invoke-ArtifactPython -Arguments @($rendererPath, (Join-Path $Stage 'draft_documents/Memoire_SMART_BUG_Exauce_Kompani.docx'), '--output_dir', (Join-Path $Stage 'render'), '--emit_pdf')
Invoke-ArtifactPython -Arguments @((Join-Path $PSScriptRoot 'check_tcn_memoire.py'), '--stage', $Stage, '--require-complete')
Write-Output 'Rendered and checked in staging. Inspect all 44 page PNGs before copying final outputs.'
