# GitHub Push 가이드

## 1. GitHub에서 새 저장소 생성

1. https://github.com/new 접속
2. Repository name: `zeroenv` (또는 원하는 이름)
3. **Public** 선택
4. **README, .gitignore 추가하지 않음** (이미 로컬에 있음)
5. Create repository

## 2. 로컬에서 Push

```powershell
cd c:\projects\zeroenv

# remote 추가 (YOUR_USERNAME을 GitHub 사용자명으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/zeroenv.git

# Push (Phase 1 커밋 + 태그)
git push -u origin master
git push origin v0.1.0-phase1
```

## 3. 버전 태그 확인

- `v0.1.0-phase1`: Phase 1 (GridWorld) 완료
- GitHub → Releases에서 태그 확인 가능
