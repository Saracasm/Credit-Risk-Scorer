# Load .env and start backend
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^([^#].+?)=(.+)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}
if ($env:GROQ_API_KEY) {
    Write-Host "GROQ_API_KEY loaded: $($env:GROQ_API_KEY.Substring(0,6))..." -ForegroundColor Green
} else {
    Write-Host "No GROQ_API_KEY found in .env" -ForegroundColor Yellow
}
& "d:\credit risk scorer\venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
