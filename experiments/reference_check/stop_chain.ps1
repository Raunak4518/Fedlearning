# Stops the reference-validation chain (chain.sh) and any reference run it started.
$pat1 = 'chain' + '.sh'
$pat2 = 'run_' + 'reference.py'
Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -and ($_.CommandLine.Contains($pat1) -or $_.CommandLine.Contains($pat2)) -and
    -not $_.CommandLine.Contains('stop_chain')
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $($_.ProcessId)" }
