# GitHub Push Guide

## 1. Create a New Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `zeroenv` (or your preferred name)
3. Select **Public**
4. **Do not add README or .gitignore** (already in local repo)
5. Create repository

## 2. Push from Local

```powershell
cd c:\projects\zeroenv

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/zeroenv.git

# Push (Phase 1 commit + tag)
git push -u origin master
git push origin v0.1.0-phase1
```

## 3. Verify Version Tags

- `v0.1.0-phase1`: Phase 1 (GridWorld) complete
- Check tags in GitHub → Releases
