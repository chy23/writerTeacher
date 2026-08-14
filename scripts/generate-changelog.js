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
                "UI：更新日誌全面中文化": { title: "UI：更新日誌全面中文化", details: ["內建英翻中對照表，將所有過去的英文更新紀錄自動轉為繁體中文顯示", "確立未來所有的系統更新紀錄將全面以繁體中文撰寫"] },
                "fix: Fetch full git history in GH actions and restore legacy changelog": { title: "修復：修正自動部署的 Git 歷史抓取限制並恢復舊版日誌", details: ["修正 GitHub Actions 預設只抓取單筆紀錄的問題，確保歷史日誌完整性", "將 V15 以前的手動更新紀錄重新合併至系統中"] },
                "chore: Remove limit on changelog generation to show all future updates": { title: "優化：解除日誌抓取筆數限制，完整呈現歷史紀錄", details: ["移除「僅顯示近50筆」的限制，確保未來數百筆更新皆能完整呈現"] },
                "feat: Auto-generate changelog from Git history": { title: "功能：新增自動從 Git 歷史產生更新日誌的功能", details: ["撰寫自動化腳本，每次部署時自動抓取 Git 歷史轉換為 JSON", "重新設計日誌對話窗介面，新增版本號、日期與 Bug 標示"] },
                "Update model fallback order": { title: "優化：更新 AI 模型備援順序", details: ["將最新模型依序設定為：3.5-flash、3.6-flash、3.7-flash、3.1-pro"] },
                "Update changelog with dates": { title: "優化：更新日誌加入確切日期", details: ["為原有的手寫更新日誌清單標註精確的發布日期"] },
                "Implement Class Roster and Image Pre-processing features": { title: "功能：實作班級長期成績資料庫與影像預處理功能", details: ["新增影像裁切與亮度調整工具，解決陰影過多或拍到桌面的問題", "新增班級成績名冊，自動歸檔並追蹤每位學生的長遠表現與評語"] },
                "Hide author signature in HTML and JS": { title: "優化：在原始碼中隱藏開發者簽名", details: ["於程式碼不影響運作的角落埋入「網站建立自楊家驊老師」的專屬宣告"] },
                "Add watermark text on top right and bottom right": { title: "功能：在圖片右上方與右下方加入浮水印", details: ["加入 25% 透明度的淺灰色浮水印文字", "採用智慧防重疊設計，自動避免浮水印遮擋核心文字"] },
                "Update model fallback order: 3-flash → 3.5-flash → 3.6-flash → 3.1-pro": { title: "優化：設定新的模型備援順序", details: ["優化 API 呼叫流程以避免單一模型擁塞失效"] },
                "Update apple-touch-icon": { title: "優化：更新手機版桌面圖示", details: ["支援 iOS 加到主畫面的專屬高解析度 App 圖示"] },
                "Add new favicon and update index.html title": { title: "功能：更新網站圖示與標題為「文本批改AI助手」", details: ["替換為極簡質感的向量網頁圖示"] },
                "Add API Key missing reminder modal with cost warning": { title: "功能：新增 API Key 設定提醒與計費警告", details: ["當未輸入金鑰即按下批改時，會自動彈出引導與費用提醒說明"] },
                "Add custom medal images to public folder": { title: "優化：新增獎牌圖示資源", details: ["將 AI 生成的高質感立體獎牌圖片正式納入專案資源庫"] },
                "Update changelog and remove version number from title": { title: "優化：更新日誌內容", details: ["移除網頁標題中的多餘版本號字串"] },
                "Use local custom PNG images as permanent default medals": { title: "功能：內建高畫質獎牌圖示為預設值", details: ["取代原有的文字或簡易圖示，全面升級等第視覺體驗"] },
                "Add built-in SVG default medals": { title: "功能：新增 SVG 獎牌圖示", details: ["提供系統備用的向量質感獎牌"] },
                "Rename app title": { title: "優化：修改網站標題", details: ["統一名稱為 WriterTeacher 文本批改助手"] },
                "Complete UI redesign for maximum layout clarity and modern aesthetics": { title: "UI：全新改版，極簡清晰的現代化介面設計", details: ["移除多餘邊框與複雜元素，改以大留白、柔和陰影與清晰的字體層級呈現"] },
                "Optimize UI contrast for better reading experience": { title: "UI：最佳化介面對比度，提升閱讀體驗", details: ["重新調整文字與背景的顏色層次", "增強按鈕與輸入框的互動反饋效果"] },
                "Configure gh-pages deployment": { title: "系統：設定 GitHub Pages 自動部署腳本", details: ["完成靜態網站打包與上線流程的自動化設定"] },
                "Enhance UI with glassmorphism and gradient background": { title: "UI：新增毛玻璃特效與漸層背景", details: ["導入 Apple 質感的半透明模糊特效"] },
                "Fix Vite build error caused by unescaped characters": { title: "修復：修正打包時的語法錯誤", details: ["解決 JSX 檔案中出現未跳脫字元導致編譯失敗的問題"] },
                "Fix GitHub Pages blank screen issue (add base URL & deploy workflow)": { title: "修復：解決 GitHub Pages 部署時的空白畫面問題", details: ["設定正確的 base URL 以對應 GitHub 網域結構"] },
                "Initial commit: Add writerTeacher source code": { title: "系統：專案建立與原始碼上傳", details: ["完成 Vite React 開發環境建置與核心架構初始化"] }
            };

            const mapping = TRANSLATIONS[subject];
            const translatedSubject = mapping ? mapping.title : subject;
            
            // Determine if this is a bug fix
            const lowerSubject = translatedSubject.toLowerCase();
            const lowerBody = body.toLowerCase();
            const isBugFix = lowerSubject.includes('fix') || lowerSubject.includes('修復') || lowerSubject.includes('解決') ||
                             lowerBody.includes('fix') || lowerBody.includes('修復') || lowerBody.includes('解決');

            // Combine git body details with our manual mapping details
            let parsedDetails = body ? body.split('\n').map(line => line.trim()).filter(line => line.length > 0) : [];
            if (mapping && mapping.details) {
                parsedDetails = [...mapping.details, ...parsedDetails];
            }

            changelog.push({
                version: `#${hash}`,
                date: date,
                title: translatedSubject,
                details: parsedDetails,
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
