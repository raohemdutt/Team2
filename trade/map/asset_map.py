import pandas as pd

# Load asset data (missing sector information)
asset_data = pd.read_csv("map/asset_data.csv")

# Load asset-to-sector mapping data
asset_sector_mapping = pd.read_csv("map/asset_sector_mappings.csv")

# Ensure column names match exactly
print("🔍 Asset Data Columns:", asset_data.columns)
print("🔍 Asset-Sector Mapping Columns:", asset_sector_mapping.columns)

# Merge based on the 'Asset' column
asset_data = asset_data.merge(asset_sector_mapping, on="Asset", how="left")

# Ensure 'Sector' column exists after merging
if "Sector" not in asset_data.columns:
    print("❌ Error: 'Sector' column is missing after merging!")
else:
    print("✅ Successfully added 'Sector' column to asset data.")

# Save the updated asset data with sectors
asset_data.to_csv("updated_asset_data_with_sectors.csv", index=False)
print("📂 Saved: updated_asset_data_with_sectors.csv")
