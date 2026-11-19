# RG routes

Builds monthly route maps for every RG cluster from the RG Data Google Sheet
and publishes them with GitHub Pages. Locations are rounded and names are hidden (--public).

Setup: add a repo secret called SHEET_URL with the sheet export link,
set Settings > Pages > Source to GitHub Actions, then run the "Build RG routes" workflow.
Fuel numbers are set in .github/workflows/build.yml (--kmpl and --fuel-price).
