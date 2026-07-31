$ErrorActionPreference = "Stop"
Start-Transcript -Path (Join-Path $PSScriptRoot "deploy_intranet.log") -Force

try {
$ruleName = "StatorYoloWeb-8001"
$displayName = "Stator YOLO Web 8001"
$creatorId = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
$subnet = "192.168.10.0/24"

if (-not (Get-NetFirewallHyperVRule -Name $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallHyperVRule `
        -Name $ruleName `
        -DisplayName $displayName `
        -Direction Inbound `
        -VMCreatorId $creatorId `
        -Protocol TCP `
        -LocalPorts 8001 `
        -RemoteAddresses $subnet `
        -Action Allow `
        -Enabled True
}

if (-not (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule `
        -Name $ruleName `
        -DisplayName $displayName `
        -Direction Inbound `
        -Protocol TCP `
        -LocalPort 8001 `
        -RemoteAddress $subnet `
        -Action Allow `
        -Enabled True
}

netsh interface portproxy delete v4tov4 `
    listenaddress=192.168.10.224 `
    listenport=8001 | Out-Null
netsh interface portproxy delete v4tov4 `
    listenaddress=0.0.0.0 `
    listenport=8001 | Out-Null
netsh interface portproxy add v4tov4 `
    listenaddress=0.0.0.0 `
    listenport=8001 `
    connectaddress=127.0.0.1 `
    connectport=8000

$taskName = "StatorYoloWeb"
$wslArguments = "-d Ubuntu-24.04 --cd /home/nina/stator-yolo .venv/bin/python run_web.py --host 0.0.0.0 --port 8000"
$action = New-ScheduledTaskAction `
    -Execute "$env:SystemRoot\System32\wsl.exe" `
    -Argument $wslArguments
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Description "Run the Stator YOLO intranet labeling website on port 8000." `
    -User $env:USERNAME `
    -RunLevel Limited `
    -Force

Start-ScheduledTask -TaskName $taskName

Write-Host "Stator YOLO intranet firewall rules installed on port 8001." -ForegroundColor Green
Write-Host "Scheduled task StatorYoloWeb installed and started." -ForegroundColor Green
}
finally {
    Stop-Transcript
}
