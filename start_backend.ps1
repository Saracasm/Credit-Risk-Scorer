# Load .env and start backend
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^([^#].+?)=(.+)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}
Write-Host "GEMINI_API_KEY loaded: $($env:GEMINI_API_KEY.Substring(0,6))..." -ForegroundColor Green
& "d:\credit risk scorer\venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
