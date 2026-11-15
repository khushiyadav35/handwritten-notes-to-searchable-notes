@echo off
echo ==============================================================
echo Pushing NoteScribe to GitHub (khushiyadav35)
echo ==============================================================
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Push failed. If the repository does not exist on GitHub yet:
    echo 1. Open your browser and go to: https://github.com/new
    echo 2. Repository Name: handwritten-notes-to-searchable-notes
    echo 3. Set visibility to: Public
    echo 4. Do NOT check README, .gitignore, or license (already created).
    echo 5. Click "Create repository"
    echo 6. Re-run this batch script!
) else (
    echo.
    echo [SUCCESS] Successfully pushed all commits to https://github.com/khushiyadav35/handwritten-notes-to-searchable-notes
    echo.
    echo [GITHUB PAGES ACTIVATION]:
    echo 1. Go to: https://github.com/khushiyadav35/handwritten-notes-to-searchable-notes/settings/pages
    echo 2. Build and deployment: GitHub Actions OR Deploy from a branch (main /docs)
    echo 3. Your live client-side web application is available at:
    echo    https://khushiyadav35.github.io/handwritten-notes-to-searchable-notes/
)
pause
