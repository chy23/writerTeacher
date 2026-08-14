import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// Get the directory of the current module
const __dirname = path.resolve();

function generateChangelog() {
    try {
        // Fetch all commits
        // Format: Hash|||Date|||Subject|||Body===END_COMMIT===
        const command = `git log --pretty=format:"%h|||%cd|||%s|||%b===END_COMMIT===" --date=short`;
        const output = execSync(command, { encoding: 'utf-8' });

        const rawCommits = output.split('===END_COMMIT===');
        
        const changelog = [];
        
        for (const rawCommit of rawCommits) {
            const commitStr = rawCommit.trim();
            if (!commitStr) continue;

            const parts = commitStr.split('|||');
            if (parts.length < 4) continue;

            const hash = parts[0].trim();
            const date = parts[1].trim();
            const subject = parts[2].trim();
            const body = parts[3].trim();

            const TRANSLATIONS = {
                "fix: Fetch full git history in GH actions and restore legacy changelog": "修復：修正自動部署的 Git 歷史抓取限制並恢復舊版日誌",
                "chore: Remove limit on changelog generation to show all future updates": "優化：解除日誌抓取筆數限制，完整呈現歷史紀錄",
                "feat: Auto-generate changelog from Git history": "功能：新增自動從 Git 歷史產生更新日誌的功能",
                "Update model fallback order": "優化：更新 AI 模型備援順序",
                "Update changelog with dates": "優化：更新日誌加入確切日期",
                "Implement Class Roster and Image Pre-processing features": "功能：實作班級長期成績資料庫與影像預處理功能",
                "Hide author signature in HTML and JS": "優化：在原始碼中隱藏開發者簽名",
                "Add watermark text on top right and bottom right": "功能：在圖片右上方與右下方加入浮水印",
                "Update model fallback order: 3-flash → 3.5-flash → 3.6-flash → 3.1-pro": "優化：設定新的模型備援順序",
                "Update apple-touch-icon": "優化：更新手機版桌面圖示",
                "Add new favicon and update index.html title": "功能：更新網站圖示與標題為「文本批改AI助手」",
                "Add API Key missing reminder modal with cost warning": "功能：新增 API Key 設定提醒與計費警告",
                "Add custom medal images to public folder": "優化：新增獎牌圖示資源",
                "Update changelog and remove version number from title": "優化：更新日誌內容",
                "Use local custom PNG images as permanent default medals": "功能：內建高畫質獎牌圖示為預設值",
                "Add built-in SVG default medals": "功能：新增 SVG 獎牌圖示",
                "Rename app title": "優化：修改網站標題",
                "Complete UI redesign for maximum layout clarity and modern aesthetics": "UI：全新改版，極簡清晰的現代化介面設計",
                "Optimize UI contrast for better reading experience": "UI：最佳化介面對比度，提升閱讀體驗",
                "Configure gh-pages deployment": "系統：設定 GitHub Pages 自動部署腳本",
                "Enhance UI with glassmorphism and gradient background": "UI：新增毛玻璃特效與漸層背景",
                "Fix Vite build error caused by unescaped characters": "修復：修正打包時的語法錯誤",
                "Fix GitHub Pages blank screen issue (add base URL & deploy workflow)": "修復：解決 GitHub Pages 部署時的空白畫面問題",
                "Initial commit: Add writerTeacher source code": "系統：專案建立與原始碼上傳"
            };

            const translatedSubject = TRANSLATIONS[subject] || subject;

            // Determine if this is a bug fix
            const lowerSubject = translatedSubject.toLowerCase();
            const lowerBody = body.toLowerCase();
            const isBugFix = lowerSubject.includes('fix') || lowerSubject.includes('修復') || lowerSubject.includes('解決') ||
                             lowerBody.includes('fix') || lowerBody.includes('修復') || lowerBody.includes('解決');

            // Filter out empty details
            const details = body ? body.split('\n').map(line => line.trim()).filter(line => line.length > 0) : [];

            changelog.push({
                version: `#${hash}`,
                date: date,
                title: translatedSubject,
                details: details,
                isBugFix: isBugFix
            });
        }

        // Write to src/changelog.json
        const targetPath = path.join(__dirname, 'src', 'changelog.json');
        fs.writeFileSync(targetPath, JSON.stringify(changelog, null, 2), 'utf-8');
        console.log(`Successfully generated changelog with ${changelog.length} entries.`);
    } catch (error) {
        console.error("Failed to generate changelog:", error.message);
        // Fallback to empty array if git fails (e.g., when not in a git repo)
        const targetPath = path.join(__dirname, 'src', 'changelog.json');
        if (!fs.existsSync(targetPath)) {
            fs.writeFileSync(targetPath, JSON.stringify([], null, 2), 'utf-8');
        }
    }
}

generateChangelog();
