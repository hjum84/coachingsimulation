import smartsheet
import os
from dotenv import load_dotenv

load_dotenv()

# Read your Smartsheet API key from environment variable
SMARTSHEET_API_KEY = os.getenv("SMARTSHEET_API_KEY")
SMARTSHEET_SHEET_ID=os.getenv("SMARTSHEET_SHEET_ID")
# Initialize Smartsheet client
smartsheet_client = smartsheet.Smartsheet(SMARTSHEET_API_KEY)
sheet = smartsheet_client.Sheets.get_sheet(SMARTSHEET_SHEET_ID)

for column in sheet.columns:
    print(f"Column Title: {column.title}, Column ID: {column.id}")