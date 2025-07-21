# PowerShell script to test demo_excel_export.py functionality
# This script addresses the PowerShell terminal issue with && operator

Write-Host "Testing demo_excel_export.py with fixed direct assignment..." -ForegroundColor Green

try {
    # Change to examples directory
    Set-Location examples
    Write-Host "✓ Changed to examples directory" -ForegroundColor Green
    
    # Run the demo_excel_export.py
    Write-Host "Running demo_excel_export.py..." -ForegroundColor Yellow
    python demo_excel_export.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ demo_excel_export.py ran successfully!" -ForegroundColor Green
        Write-Host "🎉 Direct assignment with units is now working!" -ForegroundColor Cyan
    } else {
        Write-Host "❌ demo_excel_export.py failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error occurred: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    # Return to original directory
    Set-Location ..
}

Write-Host "`nTest completed." -ForegroundColor Green