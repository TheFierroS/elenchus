# Probe: what does vcpkg + MSVC actually yield, per port?
#
# The scale estimate for route C rests on one number nobody has measured:
# how many named functions a randomly chosen vcpkg port gives us after it
# builds. This samples ports at random, builds each with MSVC in both the
# debug (/Od) and release (/O2) configurations vcpkg produces, and counts the
# procedures its PDBs declare.
#
# It answers three things and judges none of them:
#   1. what share of ports build at all, unattended;
#   2. what share of those leave usable PDBs;
#   3. the distribution of procedures per port - the number the estimate guesses.
#
# Random sampling is the point. A hand-picked list would measure our taste,
# not the ecosystem, which is exactly how the manifest got to 136 packages.
#
# Run from PowerShell on the Windows side, not WSL: vcpkg and MSVC live there.
#
#   .\probe_vcpkg_scale.ps1 -Sample 40 -OutFile vcpkg_probe.csv
#
# Needs: vcpkg on PATH, a Visual Studio build toolset, and llvm-pdbutil
# (winget install LLVM.LLVM). Expect hours and tens of gigabytes: a port may
# drag in a large dependency tree, which is why each one has a time limit.

param(
    [int]$Sample = 40,
    [int]$Seed = 0,
    [int]$TimeoutMinutes = 20,
    [string]$Triplet = "x64-windows",
    [string]$OutFile = "vcpkg_probe.csv",
    [string]$LogDir = "vcpkg_probe_logs"
)

$ErrorActionPreference = "Continue"

function Require-Tool($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Error "$name not found on PATH. $hint"
        exit 1
    }
}

Require-Tool "vcpkg" "Install vcpkg and add it to PATH."
Require-Tool "llvm-pdbutil" "winget install LLVM.LLVM"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# The full port list, then a reproducible random sample of it. The seed is
# recorded in the output so the sample can be drawn again.
Write-Host "reading the port list..."
$allPorts = (vcpkg search --x-full-desc 2>$null) |
    ForEach-Object { ($_ -split '\s+')[0] } |
    Where-Object { $_ -and $_ -notmatch '^\s*$' -and $_ -notmatch '\[' } |
    Sort-Object -Unique

Write-Host ("ports available: {0}" -f $allPorts.Count)

$random = New-Object System.Random($Seed)
$ports = $allPorts | Sort-Object { $random.Next() } | Select-Object -First $Sample

$vcpkgRoot = (Split-Path (Get-Command vcpkg).Source -Parent)
$installed = Join-Path $vcpkgRoot "installed\$Triplet"

$rows = @()

foreach ($port in $ports) {
    Write-Host ("=" * 60)
    Write-Host ("port: {0}" -f $port)

    $before = @()
    if (Test-Path $installed) {
        $before = Get-ChildItem -Path $installed -Recurse -Include *.pdb `
            -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    }

    $log = Join-Path $LogDir "$port.log"
    $started = Get-Date

    $process = Start-Process -FilePath "vcpkg" `
        -ArgumentList @("install", "${port}:${Triplet}", "--clean-after-build") `
        -NoNewWindow -PassThru -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err"

    if (-not $process.WaitForExit($TimeoutMinutes * 60 * 1000)) {
        try { $process.Kill($true) } catch {}
        $status = "timeout"
    } elseif ($process.ExitCode -eq 0) {
        $status = "built"
    } else {
        $status = "failed"
    }

    $elapsed = [int]((Get-Date) - $started).TotalSeconds

    # Only the PDBs this port added, so a dependency built earlier is not
    # counted twice. A port that brings dependencies still gets credit for
    # them the first time, which is the honest reading: that code enters the
    # corpus once.
    $after = @()
    if (Test-Path $installed) {
        $after = Get-ChildItem -Path $installed -Recurse -Include *.pdb `
            -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    }
    $new = $after | Where-Object { $before -notcontains $_ }

    $procedures = 0
    $pdbsRead = 0
    foreach ($pdb in $new) {
        $dump = & llvm-pdbutil dump --symbols $pdb 2>$null
        if ($LASTEXITCODE -eq 0 -and $dump) {
            $count = ($dump | Select-String -Pattern 'S_(G|L)PROC32' `
                -AllMatches).Count
            $procedures += $count
            $pdbsRead += 1
        }
    }

    Write-Host ("  {0}  {1}s  pdbs={2}/{3}  procedures={4}" -f `
        $status, $elapsed, $pdbsRead, $new.Count, $procedures)

    $rows += [PSCustomObject]@{
        port        = $port
        status      = $status
        seconds     = $elapsed
        pdbs_new    = $new.Count
        pdbs_read   = $pdbsRead
        procedures  = $procedures
    }

    $rows | Export-Csv -Path $OutFile -NoTypeInformation
}

Write-Host ("=" * 60)
$built = @($rows | Where-Object { $_.status -eq "built" })
$withPdb = @($built | Where-Object { $_.pdbs_read -gt 0 })
$counts = @($withPdb | ForEach-Object { $_.procedures } | Sort-Object)

Write-Host ("sampled        : {0} ports (seed {1})" -f $rows.Count, $Seed)
Write-Host ("built          : {0} ({1:P0})" -f $built.Count,
    ($built.Count / [Math]::Max($rows.Count, 1)))
Write-Host ("with pdbs      : {0} ({1:P0} of built)" -f $withPdb.Count,
    ($withPdb.Count / [Math]::Max($built.Count, 1)))

if ($counts.Count -gt 0) {
    $median = $counts[[int]($counts.Count / 2)]
    $total = ($counts | Measure-Object -Sum).Sum
    $mean = [int]($total / $counts.Count)
    Write-Host ("procedures     : median {0}, mean {1}, total {2}" -f `
        $median, $mean, $total)
    Write-Host ""
    Write-Host ("Extrapolated over {0} ports at the mean: {1:N0} procedures" -f `
        $allPorts.Count, ($mean * $allPorts.Count *
            ($withPdb.Count / [Math]::Max($rows.Count, 1))))
    Write-Host "That is BEFORE our funnel (>= 8 instructions, cross-package"
    Write-Host "dedup, >= 25 identities per package), which on the current"
    Write-Host "corpus keeps well under half. Read it as a ceiling."
}

Write-Host ""
Write-Host ("written: {0}" -f $OutFile)
