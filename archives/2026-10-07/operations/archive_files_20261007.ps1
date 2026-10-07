$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$archive = Join-Path $workspace 'archives/2026-10-07'
if (Test-Path -LiteralPath $archive) { throw 'Archive already exists; refusing to overwrite.' }
$inventory = Get-Content -LiteralPath (Join-Path $workspace 'reports/inventaire_hors_comparaison_finale_20261007.json') -Raw | ConvertFrom-Json
$plan = @()
foreach ($item in $inventory.items) {
    $section = if ($item.group -in @('cache','local_docs')) { 'local' } else { 'historique' }
    $source = [IO.Path]::GetFullPath((Join-Path $workspace $item.path))
    $target = [IO.Path]::GetFullPath((Join-Path $archive "$section/$($item.path)"))
    if (-not $source.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase) -or -not $target.StartsWith($archive + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe path: $source" }
    $entry = Get-Item -LiteralPath $source -Force
    $cursor = $entry
    while ($cursor.FullName -ne $workspace) {
        if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $($cursor.FullName)" }
        $cursor = Get-Item -LiteralPath (Split-Path -Parent $cursor.FullName) -Force
    }
    if ($entry.PSIsContainer) { throw "Expected a file: $source" }
    $plan += [pscustomobject]@{original=$item.path; archived="$section/$($item.path)"; group=$item.group; bytes=$entry.Length; sha256=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower(); action='move'}
}
New-Item -ItemType Directory -Path $archive | Out-Null
$plan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $archive 'manifest.json') -Encoding UTF8
foreach ($item in $plan) {
    $source = Join-Path $workspace $item.original
    $target = Join-Path $archive $item.archived
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $target
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $item.sha256) { throw "Hash mismatch: $target" }
    $item.original | Add-Content -LiteralPath (Join-Path $archive 'completed.txt') -Encoding UTF8
}
Write-Output "Archived and verified $($plan.Count) files."
