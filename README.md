# 🚀 Deploy Indian Market Predictor - GitHub + Free Backend Hosting

## ⚠️ Important: Understanding the Architecture

**GitHub Pages only hosts STATIC files** (HTML, CSS, JS) — it **cannot run Python**.

So the deployment has **2 parts**:
1. **Frontend (HTML)** → Deployed on **GitHub Pages** (free, static)
2. **Backend (Python Flask)** → Deployed on a free Python hosting service like **Render** or **Railway**

```
┌─────────────────────┐         ┌──────────────────────┐
│   GitHub Pages       │  API    │   Render.com          │
│   (Frontend - HTML)  │ ──────► │   (Backend - Python)  │
│   yourname.github.io │  calls  │   your-app.onrender.com│
└─────────────────────┘         └──────────────────────┘
```

---

## 📁 Step 1: Prepare Your Project Structure

Create this folder structure on your computer:

```
indian-market-predictor/
├── backend/
│   ├── app.py                 # Flask backend
│   ├── requirements.txt       # Python dependencies
│   └── Procfile                # For Render/Railway
├── docs/                       # GitHub Pages serves from /docs
│   └── index.html              # Your HTML frontend
├── .gitignore
└── README.md
```

> 💡 We use `/docs` folder because GitHub Pages can serve directly from it without extra branches.

---

## 🔧 Step 2: Deploy Backend (Python) on Render.com (FREE)

### Why Render?
- Free tier available
- Native Python support
- Auto-deploys from GitHub
- HTTPS included

### Instructions:

1. **Create a Render account**
   - Go to https://render.com
   - Sign up with GitHub

2. **Push your backend code to GitHub first** (see Step 4 below), then:

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure the service:**
   | Setting | Value |
   |---------|-------|
   | **Name** | `indian-market-predictor-api` |
   | **Root Directory** | `backend` |
   | **Environment** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app` |
   | **Instance Type** | `Free` |

5. **Click "Create Web Service"**
   - Wait 3-5 minutes for deployment
   - You'll get a URL like: `https://indian-market-predictor-api.onrender.com`

6. **Test your backend:**
   ```
   https://indian-market-predictor-api.onrender.com/api/health
   ```
   Should return JSON with status "healthy"

### ⚠️ Free Tier Limitation:
Render's free tier "sleeps" after 15 minutes of inactivity. First request after sleep takes ~30-50 seconds to wake up. This is normal!

---

## 🌐 Alternative: Deploy Backend on Railway.app

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Set **Root Directory** to `backend`
6. Railway auto-detects Python and deploys
7. Go to Settings → Generate Domain
8. Your API will be at: `https://your-app.up.railway.app`

---

## 🌐 Alternative: Deploy Backend on PythonAnywhere

1. Go to https://www.pythonanywhere.com (free tier available)
2. Upload your `app.py` and `requirements.txt`
3. Create a new Web App → Flask
4. Configure WSGI file to point to your app
5. Your API will be at: `https://yourusername.pythonanywhere.com`

---

## 📄 Step 3: Update Frontend with Your Backend URL

Open your `index.html` and find this line:

```javascript
let API_BASE_URL = localStorage.getItem('apiBaseUrl') || 'http://localhost:5000';
```

**Option A: Hardcode your backend URL (Recommended for personal use)**
```javascript
let API_BASE_URL = localStorage.getItem('apiBaseUrl') || 'https://indian-market-predictor-api.onrender.com';
```

**Option B: Keep it configurable (app has a built-in settings banner)**
- Leave as-is
- When you open the deployed site, it will show a banner asking for the backend URL
- Enter your Render/Railway URL once — it saves to browser localStorage

---

## 📦 Step 4: Push Everything to GitHub

### Initialize Git Repository

```bash
# Navigate to your project folder
cd indian-market-predictor

# Initialize git
git init

# Create .gitignore
echo "__pycache__/
*.pyc
.env
venv/
.DS_Store" > .gitignore

# Add all files
git add .

# Commit
git commit -m "Initial commit: Indian Market Predictor"

# Create a new repository on GitHub.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/indian-market-predictor.git
git branch -M main
git push -u origin main
```

---

## 🌍 Step 5: Enable GitHub Pages

1. Go to your repository on GitHub.com
2. Click **Settings** tab
3. Click **Pages** in left sidebar
4. Under "Build and deployment":
   - **Source**: Deploy from a branch
   - **Branch**: `main`
   - **Folder**: `/docs`
5. Click **Save**
6. Wait 1-2 minutes
7. Your site will be live at:
   ```
   https://YOUR_USERNAME.github.io/indian-market-predictor/
   ```

---

## ✅ Step 6: Verify Everything Works

1. **Open your GitHub Pages URL** on desktop and mobile
2. **Check backend connection status** (top of sidebar)
3. If it shows "Backend Offline":
   - Enter your Render/Railway backend URL in the config banner
   - Click "Save & Connect"
4. **Test on mobile:**
   - Open the same URL on your phone
   - Tap the ☰ menu icon to see the sidebar
   - Try switching timeframes and toggling chart options

---

## 📱 Mobile-Specific Features Added

✅ **Hamburger Menu** - Sidebar slides in/out on mobile (☰ button top-left)
✅ **Touch-Optimized Buttons** - Larger tap targets, active states instead of hover
✅ **Horizontal Scroll Chart** - Chart scrolls horizontally on small screens instead of squishing
✅ **Responsive Grid** - Indicators stack into 2 columns on mobile
✅ **Viewport Meta Tag** - Prevents unwanted zooming issues
✅ **Auto-close Menu** - Sidebar closes automatically after selecting timeframe
✅ **Configurable API URL** - No code editing needed; set backend URL from the UI

---

## 🔄 Updating Your Deployed App

Whenever you make changes:

```bash
# Make your changes to files

# Add and commit
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin main
```

- **GitHub Pages**: Updates automatically within 1-2 minutes
- **Render/Railway**: Auto-redeploys automatically when it detects the GitHub push (if auto-deploy is enabled)

---

## 🐛 Troubleshooting

### "Backend Offline" on GitHub Pages

**Cause**: CORS issue or backend sleeping (Render free tier)

**Solutions:**
1. Wait 30-60 seconds and refresh (backend waking up)
2. Verify backend URL is correct (no trailing slash)
3. Check backend logs on Render/Railway dashboard
4. Visit the `/api/health` endpoint directly to wake it up

### Mixed Content Error (HTTPS/HTTP)

**Cause**: GitHub Pages is HTTPS, but your backend URL is HTTP

**Solution**: Ensure your backend URL uses `https://` — Render and Railway provide HTTPS by default

### CORS Error in Browser Console

**Solution**: Verify the CORS configuration in `app.py` includes your GitHub Pages domain:
```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://YOUR_USERNAME.github.io", "*"]
    }
})
```

### Chart Not Rendering on Mobile

**Solution**: 
1. Clear browser cache
2. Ensure you're using the updated `index.html` with mobile canvas sizing
3. Try rotating device or refreshing

---

## 💰 Cost Summary

| Service | Cost | Limitation |
|---------|------|------------|
| **GitHub Pages** | Free forever | Static files only |
| **Render Free Tier** | Free | Sleeps after 15 min idle |
| **Railway Free Tier** | Free ($5 credit/month) | Limited hours |
| **PythonAnywhere Free** | Free | Limited CPU seconds |

For a hobby project, **Render + GitHub Pages** is the best free combination.

---

## 📝 Quick Reference: File Checklist

Before pushing to GitHub, verify you have:

- [ ] `backend/app.py` — Flask backend with `PORT` env variable support
- [ ] `backend/requirements.txt` — includes `gunicorn`
- [ ] `backend/Procfile` — contains `web: gunicorn app:app`
- [ ] `docs/index.html` — Mobile-responsive frontend
- [ ] `.gitignore` — excludes `__pycache__`, `venv`, etc.
- [ ] Frontend `API_BASE_URL` updated or config banner tested

---

## 🎉 Final URLs You'll Have

After completing all steps:

- **Frontend (Live App)**: `https://YOUR_USERNAME.github.io/indian-market-predictor/`
- **Backend (API)**: `https://indian-market-predictor-api.onrender.com`
- **Backend Health Check**: `https://indian-market-predictor-api.onrender.com/api/health`

Share your frontend URL — anyone can now access your Indian Market Predictor from any device! 📱💻