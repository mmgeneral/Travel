# auth_test

只用來「本機測試 Google 登入並取得真的 Supabase access_token」的最小頁面。

- 純 HTML + JavaScript，沒有 React / Vue / Node / npm / Vite，也沒有建置流程。
- 不修改 `user_service` 後端邏輯，也不動 Docker 設定。
- 只使用 Supabase 的 **public anon / publishable key**。

## 1. 建立本機設定檔

```powershell
copy auth_test\config.example.js auth_test\config.js
```

編輯 `auth_test\config.js`，填入：

- Supabase **Project URL**（例：`https://abcdefgh.supabase.co`）
- Supabase **public anon / publishable key**

> `config.js` 已被 `auth_test/.gitignore` 忽略，不會進版控。
> 不要放 service_role key、Google Client Secret 或 JWT secret。

## 2. Supabase 端設定（確認一次即可）

- Authentication → Providers → **Google** 已啟用（已填入 Client ID / Secret）。
- Authentication → URL Configuration：
  - Site URL：`http://localhost:3000/`
  - Redirect URLs 需包含：`http://localhost:3000/`

## 3. 啟動測試頁

```powershell
python -m http.server 3000 -d auth_test
```

打開瀏覽器：

```text
http://localhost:3000
```

## 4. 取得 access token

1. 點 **Sign in with Google**，完成 Google 授權。
2. 瀏覽器會導回 `http://localhost:3000/`，頁面會自動 `getSession()` 並顯示 `access_token`。
3. 按 **Copy access token**。

## 5. 用 token 測 user_service（PowerShell）

```powershell
$token = "PASTE_TOKEN_HERE"

$headers = @{
    Authorization = "Bearer $token"
}

Invoke-RestMethod `
  http://localhost:8001/api/v1/users/me `
  -Headers $headers
```

- 成功：回傳該使用者的 JSON（代表 Supabase JWT 驗證通過）。
- 失敗 `401`：token 過期、audience/issuer 不符，或 Redirect URL / Provider 設定未生效。

## 備註

- access token 預設約 1 小時過期；頁面已開啟 auto refresh，過期後登出再登入即可。
- 不需要這個資料夾時，直接刪掉 `auth_test\` 即可：

```powershell
Remove-Item -Recurse -Force auth_test
```
