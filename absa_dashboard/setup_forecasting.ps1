# Sentiment Forecasting Setup Script
# Run this after installing base requirements

Write-Host "🔮 Setting up Sentiment Forecasting module..." -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

Write-Host ""
Write-Host "Installing forecasting dependencies..." -ForegroundColor Yellow

# Try pip install first
try {
    pip install prophet statsmodels
    Write-Host "✅ Successfully installed via pip" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Pip install failed. Trying conda..." -ForegroundColor Yellow
    
    # Try conda install
    try {
        conda install -c conda-forge prophet statsmodels -y
        Write-Host "✅ Successfully installed via conda" -ForegroundColor Green
    } catch {
        Write-Host "❌ Installation failed. Please install manually:" -ForegroundColor Red
        Write-Host "   Option 1 (conda): conda install -c conda-forge prophet statsmodels" -ForegroundColor White
        Write-Host "   Option 2 (pip): pip install prophet statsmodels" -ForegroundColor White
        Write-Host ""
        Write-Host "Note: On Windows, conda is recommended for Prophet" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow

# Test imports
$testScript = @"
import sys
try:
    import prophet
    print(f'✅ Prophet version: {prophet.__version__}')
except ImportError as e:
    print(f'❌ Prophet import failed: {e}')
    sys.exit(1)

try:
    import statsmodels
    print(f'✅ Statsmodels version: {statsmodels.__version__}')
except ImportError as e:
    print(f'❌ Statsmodels import failed: {e}')
    sys.exit(1)

try:
    from forecasting import SentimentForecaster, PROPHET_AVAILABLE
    if PROPHET_AVAILABLE:
        print('✅ Forecasting module loaded successfully')
    else:
        print('❌ Forecasting module loaded but Prophet not available')
        sys.exit(1)
except ImportError as e:
    print(f'❌ Forecasting module import failed: {e}')
    sys.exit(1)

print('')
print('🎉 All forecasting dependencies installed successfully!')
print('')
print('Next steps:')
print('1. Run dashboard: streamlit run dashboard.py')
print('2. Navigate to 🔮 Sentiment Forecasting page')
print('3. Select an aspect and generate forecast')
print('')
print('Documentation:')
print('- Full guide: FORECASTING_GUIDE.md')
print('- Quick reference: FORECASTING_QUICK_REF.txt')
"@

python -c $testScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Setup complete! Forecasting module ready to use." -ForegroundColor Green
    Write-Host ""
    Write-Host "📚 Documentation available:" -ForegroundColor Cyan
    Write-Host "   - FORECASTING_GUIDE.md (comprehensive guide)" -ForegroundColor White
    Write-Host "   - FORECASTING_QUICK_REF.txt (quick reference)" -ForegroundColor White
    Write-Host "   - FORECASTING_IMPLEMENTATION.md (technical details)" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Setup incomplete. Please check errors above." -ForegroundColor Red
    exit 1
}
