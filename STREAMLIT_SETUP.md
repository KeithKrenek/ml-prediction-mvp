# Streamlit Cloud Deployment Setup

## Fixing the SQLite Database Error

If you're seeing this error on Streamlit Cloud:
```
OperationalError: (sqlite3.OperationalError) unable to open database file
```

This means your `DATABASE_URL` environment variable is not configured in Streamlit Cloud, so the app is falling back to SQLite (which doesn't work well on Streamlit's ephemeral filesystem).

## Solution: Configure Neon Database URL in Streamlit Secrets

### Step 1: Get Your Neon Database URL

1. Go to [Neon Console](https://console.neon.tech)
2. Select your project
3. Click on **"Connection Details"**
4. Copy the connection string (it looks like this):
   ```
   postgresql://user:password@ep-xxx-xxx.neon.tech/dbname?sslmode=require
   ```

### Step 2: Add to Streamlit Cloud Secrets

1. Go to your Streamlit Cloud dashboard: https://share.streamlit.io
2. Click on your app (**Trump Post Predictor**)
3. Click the **"Settings"** button (⚙️)
4. Click **"Secrets"** in the left sidebar
5. Add the following in the secrets text box:

```toml
DATABASE_URL = "postgresql://your-user:your-password@ep-xxx.neon.tech/your-db?sslmode=require"
```

6. Click **"Save"**
7. Your app will automatically restart with the new configuration

### Step 3: Verify It Works

- The health check endpoint should now succeed
- The app should connect to your Neon database
- No more SQLite errors!

## Alternative: Use Environment Variables (Not Recommended)

You can also set environment variables in Streamlit Cloud, but **secrets are preferred** because:
- They're encrypted
- They don't show up in logs
- They're easier to manage

## Troubleshooting

### Error: "Connection refused"
- Check that your Neon database is running
- Verify the connection string is correct
- Make sure `sslmode=require` is in the URL

### Error: "Database doesn't exist"
- Run the initialization script first to create tables
- Or connect to your Neon database and create the schema manually

### Still seeing SQLite errors?
- Clear your Streamlit Cloud cache (Settings → Clear Cache)
- Verify the secrets were saved (check for typos)
- Restart the app manually

## Need Help?

Check the [Streamlit Cloud Secrets documentation](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
