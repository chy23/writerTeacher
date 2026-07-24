import React, { useState, useRef, useEffect, useMemo } from 'react';
import { 
  FileText, 
  Upload, 
  Settings, 
  AlertCircle, 
  CheckCircle2, 
  Download, 
  Loader2, 
  Trash2, 
  User, 
  PlusCircle, 
  BarChart3, 
  RefreshCw, 
  Square, 
  Hash, 
  MousePointer2, 
  Quote, 
  ListChecks, 
  Printer, 
  FileSpreadsheet, 
  X, 
  Type, 
  WifiOff, 
  BookOpen, 
  History, 
  Cpu, 
  Trophy, 
  Medal, 
  Award, 
  Star, 
  Crown, 
  Shield, 
  ThumbsUp, 
  Sparkles, 
  Zap, 
  MessageSquareQuote, 
  Edit3, 
  PlayCircle, 
  StopCircle, 
  School, 
  Image as ImageIcon, 
  RotateCcw, 
  GraduationCap, 
  FileBadge, 
  UploadCloud, 
  DownloadCloud, 
  Eraser, 
  AlertTriangle, 
  RotateCw, 
  Search, 
  Globe, 
  Layers, 
  FileDown, 
  Wind, 
  CheckSquare, 
  Pencil, 
  Scissors, 
  Camera, 
  Database, 
  Save, 
  FileType, 
  Split, 
  Microscope, 
  Landmark, 
  Tag, 
  ShieldAlert, 
  ShieldCheck, 
  ToggleLeft, 
  ToggleRight, 
  Maximize2 
} from 'lucide-react';

// --- 1. 常數設定 (Constants - Refined based on 108 Curriculum) ---

const GRADE_LEVELS = [
  "國小低年級 (1-2 年級)",
  "國小中年級 (3-4 年級)",
  "國小高年級 (5-6 年級)",
  "國中 (7-9 年級)",
  "高中 (10-12 年級)"
];

const TEXT_TYPES = [
  { id: "auto", label: "🧬 探究文本分類 (Auto)", desc: "AI 自動判斷文本屬於「自然科學探究」或「社會領域探究」並動態切換評分規準。" },
  { id: "Ba", label: "📝 記敘文本", desc: "以人、事、時、地、物為敘寫對象，包含順敘、倒敘、插敘、補敘。" },
  { id: "Bb", label: "❤️ 抒情文本", desc: "由主體出發，抒發對人、事、物、景之情感。" },
  { id: "Bc", label: "📖 說明文本", desc: "以解釋、說明、分析、介紹為目的之文本。" },
  { id: "Bd", label: "💡 議論文本", desc: "以論述觀點、提出主張、進行辯駁為目的之文本。" },
  { id: "Bf", label: "🍃 新詩/童詩", desc: "強調意象經營、分行節奏與語言精煉度，展現想像力與真摯情感。" },
  { id: "Be", label: "✉️ 應用文本", desc: "因應日常生活或公務往來之實際需要而撰寫之文本。" }
];

const FALLBACK_MODEL_ORDER = [
  { id: 'gemini-3-flash-preview', name: 'Gemini-3-Flash' },
  { id: 'gemini-3.5-flash', name: 'Gemini-3.5-Flash' },
  { id: 'gemini-2.5-flash-preview-09-2025', name: 'Gemini-2.5-Flash' },
];

// Originality System Prompt from User
const ORIGINALITY_SYSTEM_PROMPT = `
# Role
你是一位專業的「AI 生成內容與文字原創性分析專家」。你的任務是精準辨識使用者輸入的文本中，哪些部分是人類原創、哪些具有 AI 生成特徵，以及哪些涉嫌抄襲。

# Task Workflow
請嚴格按照以下三個步驟執行分析：
1. 【文本切分】：將用戶輸入的完整文本，依據語意拆分成數個「完整的句子」或「段落」。拆分後的片段重新拼湊必須 100% 等於原文（包含標點符號與換行）。
2. 【特徵判定】：針對每一個切分出來的片段，進行獨立分析，並歸類為以下三種屬性之一（ai, plagiarism, original）。
3. 【格式化輸出】：絕對遵守指定的 JSON 格式回傳結果，不要包含任何額外的對話、解釋或 Markdown 標記語法。

# Evaluation Criteria (屬性判定標準)
針對每個片段，請依據以下標準嚴格判定：

- 類別一 "ai" (AI 生成)：
  * 具有明顯 AI 生成特徵。
  * 過度使用邏輯連接詞（例如：首先、其次、最後）。
  * 句式過於工整對稱，缺乏人類寫作的口語化波動與自然瑕疵。
  * 使用常見的 AI 慣用語（例如：「總結來說」、「值得注意的是」、「這不僅...更...」等）。

- 類別二 "plagiarism" (疑似抄襲)：
  * 高度疑似直接複製自公開網路資源。
  * 【防呆限制】：除非你非常確定這是來自知名網站（如維基百科、政府公告、新聞媒體等）的內容，否則不要輕易標記為抄襲。
  * 若標記為此類別，必須盡可能提供真實的可能來源名稱與網址。

- 類別三 "original" (人類原創)：
  * 無上述兩者特徵，語氣自然，看起來像是一般人類原創撰寫的內容。

# Output Format (嚴格 JSON 輸出)
請僅回傳以下 JSON 格式，不要加入任何 \`\`\`json 標籤或額外文字：
{
  "segments": [
    {
      "id": 0, 
      "content": "完整的原始句子或段落(一字不漏)",
      "type": "ai" | "plagiarism" | "original",
      "confidence": 85,
      "reason": "具體的判斷原因(為何覺得是AI寫的，或為何覺得是抄襲，請用繁體中文詳細說明)",
      "source": { 
        "title": "來源名稱", 
        "url": "來源真實網址" 
      }
    }
  ]
}
`;


// V13.0.0: 依據上傳的 PDF 課綱資料，精確定義探究評分向度 (用於 Canvas 繪圖與 AI Prompt)
const INQUIRY_RUBRICS_DISPLAY = {
  "Route A: 自然科學探究": [
    { 
      label: "探究與邏輯", 
      weight: "40%", 
      desc: "變因控制、假設驗證、推理論證 (代碼: tr-Vc-1, pe-Vc-2)" 
    },
    { 
      label: "證據與資料", 
      weight: "30%", 
      desc: "數據分析、圖表製作、解釋數據 (代碼: pa-Vc-2, po-Vc-2)" 
    },
    { 
      label: "表達與結構", 
      weight: "30%", 
      desc: "科學符號、報告結構、溝通表達 (代碼: pc-Vc-2, 國-表達)" 
    }
  ],
  "Route B: 社會領域探究": [
    { 
      label: "探究與邏輯", 
      weight: "40%", 
      desc: "多重觀點、時空脈絡、價值判斷 (代碼: 社1b-V-1, 歷1c-V-2)" 
    },
    { 
      label: "證據與資料", 
      weight: "30%", 
      desc: "資料蒐集、事實查證、資料判讀 (代碼: 社3a-V-1, 公3b-V-1)" 
    },
    { 
      label: "表達與結構", 
      weight: "30%", 
      desc: "論述結構、語句通順、修辭運用 (代碼: 國-表達)" 
    }
  ]
};

// V13.0.0: 詳細分級批改標準 (整合 PDF 內容)
const RUBRIC_CONTENT_ELEMENTARY = `
📂 第一部分：國小階段 (Elementary Level)
適用年級： 3-6 年級 (Learning Stages II & III)

【Type A】自然科學類寫作 (Natural Sciences)
重點：五官觀察 (自-E-A1)、圖表製作 (pa-Ⅲ-1)、區分觀察與想像 (an-Ⅲ-1)。
L5(創造): 觀察敏銳具系統性，多重感官捕捉細節。將數據轉化為精確圖表，提出新發現。
L4(分析): 觀察詳實。清楚區分「觀察到的」與「推測的」。
L3(達標): 觀察紀錄完整。使用簡單工具。文句通順。
L2(了解): 僅視覺觀察，缺乏數據。多名詞堆砌。
L1(待強): 無法區分事實與想像。

【Type B】社會領域類寫作 (Social Studies)
重點：辨別事實與意見 (1a-Ⅱ-1)、省思個人生活 (2c-Ⅱ-1)。
L5(實踐): 具行動意識。區分事實與意見，提出具體改善建議。
L4(分析): 觀點清晰。辨別事實陳述與個人意見。
L3(達標): 描述完整。舉例說明社會事物。
L2(覺察): 僅有敘述。缺乏原因思考。
L1(待強): 自我中心，混淆事實與情緒。
`;

const RUBRIC_CONTENT_JUNIOR = `
📂 第二部分：國中階段 (Junior High Level)
適用年級： 7-9 年級 (Learning Stage IV)

【Type A】自然科學類寫作 (Natural Sciences)
重點：控制變因 (pe-Ⅳ-1)、推理論證 (tr-Ⅳ-1)。
L5(評鑑): 論證嚴謹具批判性。主動評估誤差，提出改進。
L4(分析): 邏輯推論完整。連結知識與數據，推論因果。
L3(達標): 實驗步驟清楚。辨明變因。結論與數據相符。
L2(了解): 僅操作紀錄。缺乏變因控制概念。
L1(待強): 結論與證據無關，引用錯誤原理。

【Type B】社會領域類寫作 (Social Studies)
重點：多元觀點 (社 1c-Ⅳ-2)、資料適切性 (社 3b-Ⅳ-2)。
L5(批判): 系統性思辨。批判檢視資料權力關係與偏見。
L4(分析): 觀點並陳。置於時空脈絡解釋。
L3(達標): 資料運用得當。區分正反意見。
L2(了解): 單一視角。資料未查證。
L1(待強): 盲從與偏見。
`;

const RUBRIC_CONTENT_SENIOR = `
📂 第三部分：高中階段 (Senior High Level)
適用年級： 10-12 年級 (Learning Stage V)

【Type A】自然科學類寫作 (Natural Sciences)
重點：建立模型 (tm-Ⅴc-1)、批判檢核 (自S-U-A2)。
L5(創造): 建立模型解釋現象，據新證據修正。符合學術規範。
L4(評鑑): 批判反思。探討理論與實作落差原因。
L3(達標): 系統性分析。正確使用符號公式溝通。
L2(應用): 套用公式但缺乏適用範圍討論。
L1(待強): 知識碎片化，誤解科學本質。

【Type B】社會領域類寫作 (Social Studies)
重點：提出合理論證 (公 1c-Ⅴ-2)、跨科整合 (地 1c-Ⅴ-3)。
L5(轉化): 提出創新可行且符合正義的方案。高度人文關懷。
L4(評鑑): 跨領域整合。評估政策衝擊。
L3(達標): 論證嚴謹。主張、證據、推論完整。
L2(應用): 論述表面。證據薄弱。
L1(待強): 意識形態固著。無視客觀事實。
`;

// V12.1.0: 依據《108課綱：國語文領域》修正各階段學習重點
const CURRICULUM_PERFORMANCE = {
  "國小低年級 (1-2 年級)": [ "6-I-1 使用常用標點符號", "6-I-2 積累寫作材料", "6-I-3 寫出語意完整的句子", "4-I-1 認識常用國字", "5-I-4 了解文本重要訊息" ],
  "國小中年級 (3-4 年級)": [ "6-II-1 使用各種標點符號", "6-II-2 培養感受力與想像力", "6-II-3 學習審題與立意", "6-II-4 書寫記敘/說明事物", "4-II-1 認識常用國字1800字" ],
  "國小高年級 (5-6 年級)": [ "6-III-2 培養思考力與聯想力", "6-III-3 掌握寫作步驟與段落分明", "6-III-6 練習各種寫作技巧", "6-III-7 修改潤飾作品", "5-III-4 區分客觀事實與主觀判斷" ],
  "國中 (7-9 年級)": [ "6-IV-1 善用標點增進情感", "6-IV-2 結構完整/文辭優美", "6-IV-5 主動創作與闡述見解", "6-IV-6 運用資訊科技編輯", "5-IV-2 理解寫作目的與觀點" ],
  "高中 (10-12 年級)": [ "6-V-1 深化寫作能力", "6-V-4 掌握文學表現手法/關懷當代議題", "6-V-5 反覆推敲深化內涵", "6-V-6 學習多元類型創作", "5-V-2 發展系統性思考建立論述" ]
};

// V11.2.0: 教學實施重點
const CURRICULUM_GUIDELINES = {
  "國小低年級 (1-2 年級)": "寫作教學應由口述作文開始引導，著重興趣培養。強調「識字」與「寫字」的正確性，避免過度指責錯字。",
  "國小中年級 (3-4 年級)": "由口述轉換成筆述作文，引導主動寫作。強調「段落」概念（自然段轉向意義段）及完整篇章的練習。",
  "國小高年級 (5-6 年級)": "培養熟練筆述及樂於發表。強調「修辭技巧」與「潤飾」能力，並能嘗試說明與議論文本。",
  "國中 (7-9 年級)": "鼓勵藉不同文體抒發情懷，關注社會議題。強調「結構嚴謹」與「論據說服力」，並能運用資訊科技輔助。",
  "高中 (10-12 年級)": "深化溝通論述能力，符合學術與職場需求。強調「批判思考」、「個人風格」與「跨文本/跨文化」的視野。"
};

// V12.1.0: 標準國語文評分權重
const STANDARD_SCORING_RUBRICS = {
  "國小低年級 (1-2 年級)": [
    { label: "字詞與格式", weight: "40%", desc: "字體工整，筆畫正確，標點符號運用得當 (4-I-5, 6-I-1)。" },
    { label: "內容與想法", weight: "30%", desc: "能表達自我感受，內容不離題 (6-I-2)。" },
    { label: "語句通順", weight: "20%", desc: "寫出語意完整的句子 (6-I-3)。" },
    { label: "段落結構", weight: "10%", desc: "認識自然段，敘事有順序 (Ad-I-1)。" }
  ],
  "國小中年級 (3-4 年級)": [
    { label: "立意與取材", weight: "35%", desc: "主題明確，內容具豐富想像力 (6-II-3)。" },
    { label: "結構與組織", weight: "25%", desc: "能區分意義段，段落安排合理 (6-II-3)。" },
    { label: "字詞與格式", weight: "25%", desc: "正確使用常用字與標點 (4-II-1, 6-II-1)。" },
    { label: "修辭與技巧", weight: "15%", desc: "運用改寫、擴寫技巧 (6-II-6)。" }
  ],
  "國小高年級 (5-6 年級)": [
    { label: "立意與取材", weight: "35%", desc: "能提出觀點，選材適切 (6-III-2)。" },
    { label: "結構與組織", weight: "30%", desc: "篇章結構完整，段落銜接自然 (6-III-3)。" },
    { label: "修辭與技巧", weight: "20%", desc: "靈活運用修辭，練習各種寫作技巧 (6-III-6)。" },
    { label: "字詞與格式", weight: "15%", desc: "具備潤飾能力，用字精確 (6-III-7)。" }
  ],
  "國中 (7-9 年級)": [
    { label: "立意與取材", weight: "40%", desc: "自訂題目，提出見解，關注社會議題 (6-IV-5)。" },
    { label: "結構與組織", weight: "30%", desc: "結構嚴謹(起承轉合)，主旨明確 (6-IV-2)。" },
    { label: "遣詞與造句", weight: "20%", desc: "文辭優美，善用成語修辭 (6-IV-2)。" },
    { label: "字體與格式", weight: "10%", desc: "正確使用文字與標點，格式規範 (4-IV-1)。" }
  ],
  "高中 (10-12 年級)": [
    { label: "立意與取材", weight: "40%", desc: "建立論述體系，具批判哲思與當代關懷 (6-V-4)。" },
    { label: "遣詞與造句", weight: "25%", desc: "語言精煉，建立個人風格，精確運用詞彙 (6-V-5)。" },
    { label: "結構與組織", weight: "25%", desc: "結構嚴謹，論證邏輯清晰，深化內涵 (6-V-3)。" },
    { label: "字體與格式", weight: "10%", desc: "符合學術或專業格式規範，用字精準 (4-V-1)。" }
  ]
};

const POETRY_RUBRICS = [
  { label: "意象 (Imagery)", weight: "30%", desc: "運用想像力與聯想力，將抽象情感具象化。" },
  { label: "形式 (Form)", weight: "20%", desc: "透過分行控制節奏；文字精練，避免冗贅。" },
  { label: "修辭 (Rhetoric)", weight: "20%", desc: "靈活運用擬人、比喻等技巧增強表現力。" },
  { label: "內容 (Content)", weight: "30%", desc: "流露真摯情感或獨特見解（童趣或哲思）。" }
];

const POETRY_PROGRESS = {
  "國小低年級 (1-2 年級)": "聆聽與朗讀，理解詩歌訊息。",
  "國小中年級 (3-4 年級)": "學習「仿寫童詩」，掌握趣味與節奏。",
  "國小高年級 (5-6 年級)": "能夠「創作童詩」，表達觀察與想像。",
  "國中 (7-9 年級)": "新詩創作，強調意象經營與獨特風格。",
  "高中 (10-12 年級)": "深化意象、象徵手法與藝術內涵。"
};

const CHANGELOG = [
  { version: "V15.3.0", content: ["系統：調整模型優先順序，將 Gemini-3-Flash 設為首選備援模型"] },
  { version: "V15.2.0", content: ["系統：新增 Gemini-3.5-Flash 模型並設為優先預設"] },
  { version: "V15.1.0", content: ["優化：圖卡版面防溢出機制，評語自動限制最多 2 點，超出範圍自動以「...」省略", "優化：AI 提示詞更新，強制模型產出精簡評語"] },
  { version: "V15.0.2", content: ["修復：修正原創性檢測服務 API 呼叫錯誤，解決永遠顯示『暫時無法回應』的問題", "優化：增強 JSON 解析穩定度"] },
  { version: "V15.0.1", content: ["修復：增強 AI 評分表的解析能力，解決因格式不穩定導致的『無法讀取分數』錯誤"] },
  { version: "V15.0.0", content: ["重大架構更新：採用雙階段 API 呼叫，原創性分析全面升級為結構化 JSON 判定，準確度大幅提升", "UI 優化：原創性結果以直觀的百分比進度條呈現，包含 AI 佔比與抄襲佔比", "優化：Canvas 匯出圖卡完美整合全新百分比數據顯示"] },
  { version: "V14.0.0", content: ["重大更新：UI 全面優化，分數條增加黑色實線外框，確保黑白列印清晰", "修復：修正原創性檢測顯示問題與分數條數值異常", "新增：圖片瀏覽器功能"] },
  { version: "V13.23.0", content: ["UI 優化：強化分數條外框為黑色實線，確保黑白列印清晰可見"] },
];

// --- 2. 輔助函式 (Helpers) ---

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const fileToBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => resolve(reader.result);
  reader.onerror = error => reject(error);
});

// Dynamic Script Loader (for PDF/Docx/PPT parsing)
const loadScript = (src) => {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const script = document.createElement('script');
    script.src = src; script.async = true;
    script.onload = resolve; script.onerror = reject;
    document.head.appendChild(script);
  });
};

const renderTextToImage = (text, title = "Document Content") => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const w = 800; const padding = 40; const lineHeight = 24;
  ctx.font = '16px "Microsoft JhengHei", sans-serif';
  const lines = wrapText(ctx, text, w - padding * 2);
  const h = Math.max(1100, lines.length * lineHeight + padding * 3 + 60);
  canvas.width = w; canvas.height = h;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, 0, w, 80);
  ctx.fillStyle = '#334155'; ctx.font = 'bold 24px "Microsoft JhengHei", sans-serif'; ctx.fillText(title, padding, 50);
  ctx.fillStyle = '#000000'; ctx.font = '16px "Microsoft JhengHei", sans-serif';
  let y = 120;
  lines.forEach(line => { ctx.fillText(line, padding, y); y += lineHeight; });
  return new Promise(resolve => { canvas.toBlob(blob => { resolve(new File([blob], `${title}.png`, { type: 'image/png' })); }, 'image/png'); });
};

// IndexedDB Helpers
const DB_NAME = 'EssayGraderDB';
const DB_VERSION = 1;
const STORE_NAME = 'appState';

const openDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) db.createObjectStore(STORE_NAME);
    };
  });
};

const saveToDB = async (key, value) => {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put(value, key);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

const loadFromDB = async (key) => {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const clearDB = async () => {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.clear();
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

const loadJsPDF = () => {
  return new Promise((resolve, reject) => {
    if (window.jspdf) { resolve(window.jspdf); return; }
    const script = document.createElement('script');
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
    script.async = true;
    script.onload = () => resolve(window.jspdf);
    script.onerror = () => reject(new Error("無法載入 PDF 生成組件"));
    document.head.appendChild(script);
  });
};

const getGradeInfo = (score) => {
  const s = parseInt(score);
  if (isNaN(s)) return { label: "待評", color: "text-slate-400", type: "none" };
  if (s >= 93) return { label: "特優", color: "text-purple-600", type: "grand" };
  if (s >= 85) return { label: "優等", color: "text-rose-600", type: "merit" };
  if (s >= 80) return { label: "金獎", color: "text-amber-500", type: "gold" };
  if (s >= 75) return { label: "銀獎", color: "text-slate-500", type: "silver" };
  if (s >= 70) return { label: "佳作", color: "text-orange-600", type: "good" };
  return { label: "普獎", color: "text-blue-600", type: "pass" };
};

const getGradeColorHex = (type) => {
  switch (type) {
    case 'grand': return '#9333ea';
    case 'merit': return '#e11d48';
    case 'gold': return '#d97706';
    case 'silver': return '#64748b';
    case 'good': return '#ea580c';
    case 'pass': return '#2563eb';
    default: return '#94a3b8';
  }
};

const stripCodes = (text) => {
  if (!text) return "";
  let cleaned = text;

  // 1. Remove bracketed codes (broad pattern for safety)
  // Matches ( ... 1a-III-1 ... ) or ( ... 自-E-A1 ... )
  cleaned = cleaned.replace(/[\(（\[].*?[\)）\]]/g, (match) => {
      // Check if the content looks like a curriculum code
      if (match.match(/-[IVXivxⅠⅡⅢⅣⅤ]+-/)) return ""; // Roman numerals structure
      if (match.match(/[自社國]-[EJS]-[A-C]\w+/)) return ""; // Core Competency
      if (match.match(/[環性人道家品法科資能安防戶國原]+[A-Z]\d+/)) return ""; // Issues
      return match;
  });

  // 2. Remove naked codes (Powerful Regex for all formats)
  // Format: 社 1a-Ⅲ-1, 1a-Ⅲ-1 (Content/Performance)
  cleaned = cleaned.replace(/(?:[社公歷地國自然數健藝綜活]\s*)?[a-zA-Z0-9]+\s*-[IVXivxⅠⅡⅢⅣⅤ]+-\d+/gi, "");
  
  // Format: 自-E-A1 (Core Competency)
  cleaned = cleaned.replace(/[自社國語數健藝綜活科]-[EJS]-[A-C]\w+/gi, "");
  
  // Format: 環E1, 性U1 etc (Issue Infusion)
  cleaned = cleaned.replace(/[環性人道家品法科資能安防戶國原]+[A-Z]\d+/gi, "");

  // Format: English codes like tr-IV-2, ai-IV-3, pa-III-1 (Inquiry)
  cleaned = cleaned.replace(/[a-zA-Z]{2,}-[IVXivxⅠⅡⅢⅣⅤ]+-\d+/gi, "");

  // Format: tr-Vc-1 (High school inquiry)
  cleaned = cleaned.replace(/[a-zA-Z]{2,}-[A-Za-z]+-\d+/gi, "");

  // Cleanup extra spaces and punctuation left behind
  return cleaned.replace(/\s{2,}/g, " ").replace(/ ,/g, ",").trim();
};

// --- 4. 繪圖工具 (Drawing Utils) ---

const wrapText = (ctx, text, maxWidth) => {
  if (!text) return [];
  const words = text.split('');
  const lines = [];
  let currentLine = words[0] || "";
  for (let i = 1; i < words.length; i++) {
    const word = words[i];
    const width = ctx.measureText(currentLine + word).width;
    if (width < maxWidth) { currentLine += word; } 
    else { lines.push(currentLine); currentLine = word; }
  }
  lines.push(currentLine);
  return lines;
};

const drawBulletText = (ctx, text, x, y, maxWidth, lineHeight, color = "#334155", maxY = 9999, dotScale = 1) => {
  if (!text) return y;
  const rawLines = text.split('\n').filter(l => l.trim().length > 0);
  let currentY = y;
    
  for (let i = 0; i < rawLines.length; i++) {
    if (currentY + lineHeight > maxY) break; 
    
    const rawLine = rawLines[i];
    const cleanLine = rawLine.replace(/^[\s\-\–\—•·*]+/, '').trim();
    
    ctx.fillStyle = "#6366f1";
    ctx.beginPath();
    ctx.arc(x - (20 * dotScale), currentY - (10 * dotScale), 6 * dotScale, 0, Math.PI * 2); 
    ctx.fill();
    ctx.fillStyle = color;
    
    const wrappedLines = wrapText(ctx, cleanLine, maxWidth);
    
    for (let j = 0; j < wrappedLines.length; j++) {
        if (currentY + lineHeight > maxY) break; 
        
        let textToDraw = wrappedLines[j];
        
        // 限制檢查：如果下一行會超過最大高度，強制在此行截斷並補上「...」
        if (currentY + lineHeight * 2 > maxY && (j < wrappedLines.length - 1 || i < rawLines.length - 1)) {
            textToDraw = textToDraw.length > 3 ? textToDraw.substring(0, textToDraw.length - 3) + "..." : textToDraw + "...";
            ctx.fillText(textToDraw, x, currentY);
            return maxY; // 到底了，完全停止繪製
        }
        
        ctx.fillText(textToDraw, x, currentY);
        currentY += lineHeight;
    }
    currentY += 12 * dotScale; 
  }
  return currentY;
};

// --- 3. 子元件 (Sub-Components) ---

const MedalSettings = ({ medals, onUpload, onReset }) => {
  const grades = [
    { key: 'grand', label: '特優' }, { key: 'merit', label: '優等' }, 
    { key: 'gold', label: '金獎' }, { key: 'silver', label: '銀獎' },
    { key: 'good', label: '佳作' }, { key: 'pass', label: '普獎' },
  ];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2">
        {grades.map((grade) => (
          <label key={grade.key} className="cursor-pointer group relative">
            <div className={`aspect-square rounded-xl border-2 border-dashed flex flex-col items-center justify-center transition-all overflow-hidden ${medals[grade.key] ? 'border-indigo-500 bg-white' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'}`}>
              {medals[grade.key] ? (
                <img src={medals[grade.key]} alt={grade.label} className="w-full h-full object-contain p-1" />
              ) : (
                <>
                  <ImageIcon size={16} className="text-slate-400 mb-1 group-hover:text-indigo-400" />
                  <span className="text-[10px] font-bold text-slate-500 group-hover:text-indigo-600">{grade.label}</span>
                </>
              )}
            </div>
            <input type="file" className="hidden" accept="image/*" onChange={(e) => { if (e.target.files[0]) onUpload(grade.key, e.target.files[0]); }} />
          </label>
        ))}
      </div>
      <button onClick={onReset} className="w-full py-2 text-xs text-slate-400 hover:text-rose-500 flex items-center justify-center gap-1 transition-colors">
        <RotateCcw size={12} /> 重置所有圖片
      </button>
    </div>
  );
};

const ScoreDistributionChart = ({ essays }) => {
  const stats = useMemo(() => {
    const counts = { grand: 0, merit: 0, gold: 0, silver: 0, good: 0, pass: 0 };
    essays.forEach(e => {
      const info = getGradeInfo(e.score);
      if (info.type !== 'none' && counts[info.type] !== undefined) counts[info.type]++;
    });
    return Object.entries(counts).map(([key, val]) => ({ 
      key, val, label: getGradeInfo(key === 'grand' ? 95 : key === 'merit' ? 90 : key === 'gold' ? 82 : key === 'silver' ? 77 : key === 'good' ? 72 : 60).label 
    }));
  }, [essays]);

  const maxVal = Math.max(...stats.map(s => s.val), 1);

  return (
    <div className="mt-6 p-6 bg-white rounded-2xl border border-slate-100 shadow-sm">
      <h4 className="text-sm font-bold text-slate-700 mb-6 flex items-center gap-2">
        <BarChart3 size={16} className="text-indigo-500" /> 成績分佈統計
      </h4>
      <div className="flex items-end justify-between h-32 gap-2">
        {stats.map((s, i) => (
          <div key={i} className="flex-1 flex flex-col items-center gap-2">
            <div className="text-[10px] font-bold text-slate-400">{s.val > 0 ? s.val : ''}</div>
            <div className="w-full rounded-t-lg transition-all duration-1000 bg-indigo-50/10 border-t-2 border-indigo-500" style={{ height: `${(s.val / maxVal) * 100}%`, minHeight: s.val > 0 ? '4px' : '0px' }} />
            <div className="text-[10px] font-bold text-slate-500 whitespace-nowrap">{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const SimpleMarkdown = ({ text }) => {
  if (!text) return null;
  return (
    <div className="space-y-2 font-serif">
      {text.split('\n').map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={index} className="h-2" />;
        if (line.match(/^#{1,6}\s/)) {
          return <h3 key={index} className="text-lg font-bold text-indigo-900 mt-4 mb-2">{line.replace(/^#{1,6}\s+/, '')}</h3>;
        }
        if (line.match(/^[\*\-]\s/)) {
          return (
            <div key={index} className="flex items-start gap-2 ml-2 mb-1">
              <span className="mt-2 w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
              <p className="text-slate-700 leading-relaxed">{line.replace(/^[\*\-]\s+/, '')}</p>
            </div>
          );
        }
        return <p key={index} className="text-slate-700 leading-relaxed">{line}</p>;
      })}
    </div>
  );
};

const Toast = ({ message, type, onClose, action }) => (
  <div className={`fixed bottom-6 right-6 z-50 flex items-center space-x-3 px-6 py-4 rounded-2xl shadow-2xl border animate-in slide-in-from-right duration-300 ${
    type === 'error' ? 'bg-red-50 border-red-100 text-red-600' : 'bg-white border-slate-100 text-slate-600'
  }`}>
    {type === 'error' ? <AlertCircle size={20} /> : <CheckCircle2 size={20} className="text-emerald-500" />}
    <span className="font-bold text-sm">{message}</span>
    {action && (
        <button onClick={action.onClick} className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-xs hover:bg-indigo-700 ml-2">
            {action.label}
        </button>
    )}
    <button onClick={onClose} className="p-1 hover:bg-black/5 rounded-lg transition-colors"><X size={16} /></button>
  </div>
);

// New Component: Image Viewer Modal
const ImageViewer = ({ src, onClose }) => {
  if (!src) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm animate-in fade-in duration-300" onClick={onClose}>
      <div className="relative max-w-5xl w-full max-h-[90vh] p-4 flex flex-col items-center">
        <button onClick={onClose} className="absolute top-4 right-4 p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors">
          <X size={24} />
        </button>
        <img src={src} alt="Full View" className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl" onClick={e => e.stopPropagation()} />
        <div className="mt-4 text-white/70 text-sm font-bold flex items-center gap-2">
            <Maximize2 size={14} /> 點擊背景關閉預覽
        </div>
      </div>
    </div>
  );
};

// --- 5. 主程式 (Main App) ---

const App = () => {
  const [apiKey, setApiKey] = useState('');
  const [gradeLevel, setGradeLevel] = useState(GRADE_LEVELS[2]); // Default to 5-6th grade
  const [textTypeId, setTextTypeId] = useState(TEXT_TYPES[0].id); // Default to Auto
  const [gradingDate, setGradingDate] = useState(new Date().toISOString().split('T')[0]);
  const [essayTopic, setEssayTopic] = useState('多元寫作'); 
  const [checkOriginality, setCheckOriginality] = useState(true); // New Toggle State
  const [essays, setEssays] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [toast, setToast] = useState(null);
  const [classAnalysis, setClassAnalysis] = useState('');
  const [showChangelog, setShowChangelog] = useState(false);
  const [customMedals, setCustomMedals] = useState({ grand: null, merit: null, gold: null, silver: null, good: null, pass: null });
  const [batchResult, setBatchResult] = useState(null);
  const [showRegradeConfirm, setShowRegradeConfirm] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [isLoadingFile, setIsLoadingFile] = useState(false);
  const [viewingImage, setViewingImage] = useState(null); // State for Image Viewer

  const stopRef = useRef(false);
  const abortControllerRef = useRef(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null); 
  const restoreInputRef = useRef(null);
  const essaysRef = useRef(essays);

  useEffect(() => { essaysRef.current = essays; }, [essays]);

  // Logic for default topic handling based on mode
  useEffect(() => {
    setEssayTopic(prev => {
      if (textTypeId === 'auto') {
        return prev || '多元寫作';
      } else {
        return prev === '多元寫作' ? '' : prev;
      }
    });
  }, [textTypeId]);

  // Persistence Logic
  useEffect(() => {
    const checkRecovery = async () => {
        try {
            const savedState = await loadFromDB('currentState');
            if (savedState && savedState.essays && savedState.essays.length > 0) {
                setToast({ 
                    msg: "發現未完成的批改紀錄", 
                    type: "info",
                    action: { label: "還原", onClick: () => restoreState(savedState) }
                });
            }
        } catch (e) { console.error("DB Error", e); }
    };
    checkRecovery();
  }, []);

  useEffect(() => {
    const save = setTimeout(async () => {
        if (essays.length > 0) {
             const state = { gradeLevel, textTypeId, gradingDate, essayTopic, classAnalysis, essays: essays.map(e => ({...e, images: e.images.map(img => ({...img, preview: null}))})) };
             await saveToDB('currentState', state);
        }
    }, 2000);
    return () => clearTimeout(save);
  }, [essays, gradeLevel, textTypeId, gradingDate, essayTopic, classAnalysis]);

  // Persistence for Custom Medals - Using IndexedDB
  useEffect(() => {
    const saveMedals = async () => {
         const hasCustom = Object.values(customMedals).some(v => v !== null);
         if (hasCustom) {
             try {
                 await saveToDB('customMedals', customMedals);
             } catch (e) {
                 console.error("Failed to save medals to DB", e);
             }
         }
    };
    saveMedals();
  }, [customMedals]);

  const restoreState = (savedState) => {
      setGradeLevel(savedState.gradeLevel);
      setTextTypeId(savedState.textTypeId);
      setGradingDate(savedState.gradingDate);
      setEssayTopic(savedState.essayTopic);
      setClassAnalysis(savedState.classAnalysis || '');
      const restoredEssays = savedState.essays.map(e => ({
          ...e,
          images: e.images.map(img => ({ ...img, preview: URL.createObjectURL(img.file) }))
      }));
      setEssays(restoredEssays);
      setToast(null);
  };

  // Load custom medals from IndexedDB on startup
  useEffect(() => {
    const loadMedals = async () => {
        try {
            const saved = await loadFromDB('customMedals');
            if (saved) {
                setCustomMedals(saved);
            } else {
                 const legacy = localStorage.getItem('essay_grader_medals');
                 if (legacy) { 
                     try { setCustomMedals(JSON.parse(legacy)); } catch (e) { console.error(e); } 
                 }
            }
        } catch (e) { console.error("Error loading medals", e); }
    };
    loadMedals();
  }, []);

  const showToast = (msg, type = 'info', action = null) => {
    setToast({ msg: String(msg), type, action });
    if (!action) setTimeout(() => setToast(null), 4000);
  };

  const handleMedalUpload = (key, file) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setCustomMedals(prev => { return { ...prev, [key]: reader.result }; });
      showToast(`已更新 ${key} 獎牌圖示`);
    };
    reader.readAsDataURL(file);
  };

  const handleScoreChange = (essayId, scoreIndex, newScore) => {
    setEssays(prev => prev.map(e => {
        if (e.id !== essayId) return e;
        const newDetailed = [...e.detailedScores];
        newDetailed[scoreIndex].score = parseInt(newScore);
        const newTotal = Math.min(100, newDetailed.reduce((acc, curr) => acc + curr.score, 0));
        return { ...e, detailedScores: newDetailed, score: newTotal.toString() };
    }));
  };

  const handleTextChange = (essayId, field, value) => {
      setEssays(prev => prev.map(e => e.id === essayId ? { ...e, [field]: value } : e));
  };

  const handleExportCSV = () => {
      if (essays.length === 0) return showToast("目前沒有資料可以匯出", "error");
      const BOM = "\uFEFF"; 
        
      const firstCompleted = essays.find(e => e.status === 'completed' && e.detailedScores);
      let rubricHeaders = [];
       
      if (firstCompleted) {
        rubricHeaders = firstCompleted.detailedScores.map(d => d.label);
      } else if (textTypeId === 'auto') {
         rubricHeaders = INQUIRY_RUBRICS_DISPLAY["Route A: 自然科學探究"].map(r => r.label);
      } else if (textTypeId === 'Bf') {
         rubricHeaders = POETRY_RUBRICS.map(r => r.label);
      } else {
         rubricHeaders = (STANDARD_SCORING_RUBRICS[gradeLevel] || []).map(r => r.label);
      }

      const headers = ["座號", "姓名", "總分", ...rubricHeaders, "評語摘要(亮點/建議)", "AI 佔比", "抄襲佔比", "原創性評估"];
      
      const rows = essays.map(e => {
          const scoreMap = {};
          if (e.detailedScores) { e.detailedScores.forEach(ds => scoreMap[ds.label] = ds.score); }
          const scores = rubricHeaders.map(label => scoreMap[label] || "0");
          const comments = `[亮點] ${e.highlights || ''} \n[建議] ${e.suggestions || ''}`.replace(/"/g, '""'); 

          // Originality Info for CSV
          const aiRisk = e.originality ? `${e.originality.aiRatio}%` : "N/A";
          const copyRisk = e.originality ? `${e.originality.copyRatio}%` : "N/A";
          const reason = e.originality ? e.originality.reason : "";

          return [ `"${e.studentNumber || ''}"`, `"${e.studentName || ''}"`, e.score, ...scores, `"${comments}"`, `"${aiRisk}"`, `"${copyRisk}"`, `"${reason}"` ].join(",");
      });

      const csvContent = BOM + headers.join(",") + "\n" + rows.join("\n");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${essayTopic || '成績單'}_${gradingDate}.csv`;
      link.click();
      URL.revokeObjectURL(url);
      showToast("成績單匯出成功");
  };

  const handleExportBackup = async () => {
    if (essays.length === 0) return showToast("目前沒有資料可以匯出", "error");
    setIsProcessing(true);
    try {
      const backupEssays = await Promise.all(essays.map(async (e) => {
        const imagesWithBase64 = await Promise.all(e.images.map(async (img) => ({
          base64: await fileToBase64(img.file),
          type: img.file.type,
          name: img.file.name
        })));
        return { ...e, images: imagesWithBase64 };
      }));

      const dataToExport = {
        version: "V15.0",
        timestamp: new Date().toISOString(),
        gradeLevel, textTypeId, gradingDate, essayTopic, classAnalysis, customMedals,
        essays: backupEssays
      };

      const blob = new Blob([JSON.stringify(dataToExport)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${essayTopic || '批改資料'}_備份_${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      URL.revokeObjectURL(url);
      showToast("批改紀錄已成功導出");
    } catch (err) { showToast("導出失敗", "error"); } finally { setIsProcessing(false); }
  };

  const handleImportBackup = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const data = JSON.parse(event.target.result);
        setGradeLevel(data.gradeLevel || GRADE_LEVELS[2]);
        setTextTypeId(data.textTypeId || TEXT_TYPES[0].id);
        setEssayTopic(data.essayTopic || "");
        setGradingDate(data.gradingDate || new Date().toISOString().split('T')[0]);
        if (data.classAnalysis) setClassAnalysis(data.classAnalysis);
        if (data.customMedals) { setCustomMedals(data.customMedals); localStorage.setItem('essay_grader_medals', JSON.stringify(data.customMedals)); }

        const restoredEssays = await Promise.all(data.essays.map(async (e) => {
          const restoredImages = await Promise.all(e.images.map(async (img) => {
            const res = await fetch(img.base64);
            const blob = await res.blob();
            const fileObj = new File([blob], img.name, { type: img.type });
            return { file: fileObj, preview: URL.createObjectURL(fileObj) };
          }));
          return { ...e, images: restoredImages };
        }));
        setEssays(restoredEssays);
        showToast(`成功匯入 ${restoredEssays.length} 份紀錄`);
      } catch (err) { showToast("匯入失敗，格式錯誤", "error"); }
    };
    reader.readAsText(file);
  };

  const generateSummaryPages = (analysis, grade, topic, date, essayList) => {
    const scale = 2; const w = 595 * scale; const h = 842 * scale;
    const marginX = 50 * scale; const marginY = 50 * scale; const contentWidth = w - (marginX * 2);
    const pages = [];
    const createPage = () => {
      const cvs = document.createElement('canvas');
      cvs.width = w; cvs.height = h;
      const c = cvs.getContext('2d');
      c.fillStyle = "#ffffff"; c.fillRect(0, 0, w, h);
      return { cvs, c };
    };
    let { cvs, c } = createPage();
    
    // Header
    c.fillStyle = "#4f46e5"; c.fillRect(0, 0, w, 120 * scale);
    c.fillStyle = "#ffffff"; c.font = `bold ${36 * scale}px "Microsoft JhengHei", sans-serif`; c.textAlign = "center";
    c.fillText("全班教學總結分析報告", w / 2, 75 * scale);

    // Metadata
    const metaStartY = 160 * scale; const lineHeightMeta = 30 * scale;
    c.fillStyle = "#334155"; c.font = `bold ${16 * scale}px "Microsoft JhengHei", sans-serif`; c.textAlign = "left";
    c.fillText(`班級年級：${grade}`, marginX, metaStartY);
    c.fillText(`作文題目：${topic}`, marginX, metaStartY + lineHeightMeta);
    c.fillText(`批改日期：${date}`, marginX, metaStartY + lineHeightMeta * 2);
    c.fillText(`批改份數：${essayList.length} 份`, marginX, metaStartY + lineHeightMeta * 3);

    // Stats Chart
    const stats = { grand: 0, merit: 0, gold: 0, silver: 0, good: 0, pass: 0 };
    essayList.forEach(e => { const i = getGradeInfo(e.score); if (i.type !== 'none') stats[i.type]++; });
    const maxVal = Math.max(...Object.values(stats), 1);
    
    const chartX = w - marginX - (220 * scale); const chartY = metaStartY - 10 * scale;
    const barWidth = 24 * scale; const gap = 12 * scale;

    Object.entries(stats).forEach(([k, v], idx) => {
        const info = getGradeInfo(k === 'grand' ? 95 : k === 'merit' ? 90 : k === 'gold' ? 82 : k === 'silver' ? 77 : k === 'good' ? 72 : 60);
        const barHeight = (v / maxVal) * (80 * scale);
        const xPos = chartX + idx * (barWidth + gap);
        const baseLine = chartY + 100 * scale;
        
        c.fillStyle = getGradeColorHex(k);
        c.fillRect(xPos, baseLine - barHeight, barWidth, barHeight);
        c.fillStyle = "#64748b"; c.font = `bold ${12 * scale}px "Microsoft JhengHei", sans-serif`; c.textAlign = "center";
        c.fillText(v, xPos + barWidth/2, baseLine - barHeight - 5 * scale);
        c.font = `${10 * scale}px "Microsoft JhengHei", sans-serif`;
        c.fillText(info.label, xPos + barWidth/2, baseLine + 15 * scale);
    });

    const dividerY = 300 * scale;
    c.strokeStyle = "#e2e8f0"; c.lineWidth = 2 * scale;
    c.beginPath(); c.moveTo(marginX, dividerY); c.lineTo(w - marginX, dividerY); c.stroke();

    let currentY = dividerY + 40 * scale;
    const cleanText = analysis.replace(/#{1,6}\s?/g, '').replace(/\*\*/g, '');
    const paragraphs = cleanText.split('\n');

    c.textAlign = "left";
    const HEADER_SIZE_PX = 40; const BODY_SIZE_PX = 32;      
    const HEADER_LH = 60; const BODY_LH = 48;            

    paragraphs.forEach(para => {
        if (!para.trim()) return;
        let fontSize = BODY_SIZE_PX; let fontWeight = "normal"; let color = "#334155"; let lineHeight = BODY_LH;
        const isListHeader = /^[一二三四五12345]、|\d+\./.test(para.trim());
        const isKeywordHeader = (para.length < 50 && (para.includes("整體") || para.includes("建議") || para.includes("缺點") || para.includes("表現")));

        if (isListHeader || isKeywordHeader) { fontSize = HEADER_SIZE_PX; color = "#1e293b"; fontWeight = "bold"; lineHeight = HEADER_LH; }

        c.font = `${fontWeight} ${fontSize}px "Microsoft JhengHei", sans-serif`; c.fillStyle = color;
        const lines = wrapText(c, para, contentWidth);

        lines.forEach(line => {
            if (currentY + lineHeight > h - marginY) {
                pages.push(cvs.toDataURL('image/png'));
                ({ cvs, c } = createPage());
                currentY = marginY + 20 * scale;
                c.font = `${fontWeight} ${fontSize}px "Microsoft JhengHei", sans-serif`; c.fillStyle = color;
            }
            c.fillText(line, marginX, currentY);
            currentY += lineHeight;
        });
    });
    
    pages.push(cvs.toDataURL('image/png'));
    return pages;
  };

  const drawEssayCard = async (essay, mode = 'teacher') => {
    const isA5 = textTypeId === 'auto'; // Inquiry mode = A5
    const canvas = document.createElement('canvas');
    
    // A5: 1748 x 2480 (300DPI) | A6: 1240 x 1748 (300DPI)
    const w = isA5 ? 1748 : 1240; 
    const h = isA5 ? 2480 : 1748;
    
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Layout Configuration based on size
    const layout = {
        margin: isA5 ? 30 : 20,
        headerHeight: isA5 ? 260 : 200,
        titleSize: isA5 ? 80 : 60,
        titleY: isA5 ? 160 : 120,
        metaSize: isA5 ? 48 : 36,
        metaY: isA5 ? 190 : 140,
        studentInfoSize: isA5 ? 60 : 44,
        medalY: isA5 ? 450 : 380,
        medalRadius: isA5 ? 140 : 100,
        scoreSize: isA5 ? 160 : 120,
        barStartY: isA5 ? 360 : 280,
        barGap: isA5 ? 140 : 100,
        barLabelSize: isA5 ? 40 : 30,
        commentsStartY: isA5 ? 1050 : 800, // A6 moves comments up significantly
        commentTitleSize: isA5 ? 56 : 40,
        commentBodySize: isA5 ? 44 : 32,
        commentLineHeight: isA5 ? 65 : 48,
        footerY: h - (isA5 ? 80 : 60),
        footerSize: isA5 ? 36 : 28
    };

    // --- Background & Border ---
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 4;
    ctx.strokeRect(layout.margin, layout.margin, w - (layout.margin*2), h - (layout.margin*2));

    // --- Header ---
    ctx.fillStyle = "#4f46e5";
    ctx.fillRect(layout.margin, layout.margin, w - (layout.margin*2), layout.headerHeight); 

    // Title
    ctx.fillStyle = "#ffffff"; 
    ctx.font = `bold ${layout.titleSize}px "Microsoft JhengHei", sans-serif`;
    ctx.fillText(`『${essayTopic || '多元寫作'}』評分表`, layout.margin + 50, layout.titleY); 

    // Metadata Tag
    let metaText = "";
    let metaIcon = "";
    if (essay.evaluatedType) {
        metaIcon = essay.evaluatedType.includes("自然") ? "🧪" : "⚖️";
        const typeName = essay.evaluatedType.includes("自然") ? "自然科學類" : "社會領域類";
        metaText = `${metaIcon} ${typeName}`;
    } else if (textTypeId !== 'auto') {
        const gradeText = gradeLevel.split(" ")[0];
        const textType = TEXT_TYPES.find(t=>t.id===textTypeId);
        const label = textType ? textType.label.split(" ")[1] : "一般";
        metaText = `${gradeText} · ${label}`;
    }

    if (metaText) {
        ctx.font = `bold ${layout.metaSize}px "Microsoft JhengHei", sans-serif`;
        const textMetrics = ctx.measureText(metaText);
        const tagWidth = textMetrics.width + (isA5 ? 60 : 40);
        const tagHeight = isA5 ? 80 : 60;
        const tagX = layout.margin + 50; const tagY = layout.metaY;
        
        ctx.fillStyle = "#ffffff"; // White background
        ctx.beginPath(); ctx.roundRect(tagX, tagY, tagWidth, tagHeight, 20); ctx.fill();
        
        ctx.fillStyle = "#4f46e5"; // Purple text
        ctx.fillText(metaText, tagX + (isA5 ? 30 : 20), tagY + (isA5 ? 56 : 42));
    }

    // Student Info
    ctx.textAlign = "right"; ctx.font = `bold ${layout.studentInfoSize}px "Microsoft JhengHei", sans-serif`; ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    
    let studentInfo = "";
    if (mode === 'student') {
        studentInfo = essay.studentNumber ? `#${essay.studentNumber}` : '';
    } else {
        studentInfo = essay.studentNumber ? `#${essay.studentNumber} ${essay.studentName}` : essay.studentName;
    }

    ctx.fillText(studentInfo, w - layout.margin - 50, layout.titleY); 
    ctx.textAlign = "left";

    // --- Medal & Total Score (Right Side) ---
    let displayScore = essay.score;
    if (isNaN(parseInt(essay.score)) && essay.detailedScores?.length > 0) {
        const sum = essay.detailedScores.reduce((acc, cur) => acc + (cur.score || 0), 0);
        displayScore = sum.toString();
    }
    
    const info = getGradeInfo(displayScore);
    const gradeColor = getGradeColorHex(info.type);
    const centerX_Right = w - (isA5 ? 280 : 200);
    const centerY_Medal = layout.medalY;

    // Medal Icon
    if (customMedals[info.type]) {
      const medalImg = new Image(); medalImg.src = customMedals[info.type];
      await new Promise(r => medalImg.onload = r);
      const imgSize = layout.medalRadius * 2.2;
      ctx.drawImage(medalImg, centerX_Right - imgSize/2, centerY_Medal - imgSize/2, imgSize, imgSize);
    } else {
      ctx.fillStyle = "#f1f5f9"; ctx.beginPath(); ctx.arc(centerX_Right, centerY_Medal, layout.medalRadius, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#cbd5e1"; ctx.font = `bold ${layout.medalRadius * 0.7}px "Microsoft JhengHei", sans-serif`; ctx.textAlign = "center";
      ctx.fillText("🏅", centerX_Right, centerY_Medal + (isA5 ? 30 : 20)); ctx.textAlign = "left";
    }

    // Grade Label Badge
    ctx.save();
    ctx.fillStyle = gradeColor;
    const badgeW = isA5 ? 240 : 180; const badgeH = isA5 ? 80 : 60;
    const badgeX = centerX_Right - badgeW / 2; const badgeY = centerY_Medal + layout.medalRadius + 20; 
    ctx.beginPath(); ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 30); ctx.fill();
    ctx.fillStyle = "#ffffff"; ctx.font = `bold ${layout.metaSize}px "Microsoft JhengHei", sans-serif`;
    ctx.textAlign = "center"; ctx.fillText(info.label, centerX_Right, badgeY + (isA5 ? 56 : 42));
    ctx.restore();

    // Big Score
    if (mode === 'teacher') {
      ctx.fillStyle = "#4f46e5"; ctx.textAlign = "center";
      ctx.font = `bold ${layout.scoreSize}px "Microsoft JhengHei", sans-serif`; 
      const scoreY = badgeY + (isA5 ? 220 : 160);
      ctx.fillText(displayScore, centerX_Right, scoreY); 
      ctx.font = `bold ${layout.metaSize}px "Microsoft JhengHei", sans-serif`; ctx.fillStyle = "#94a3b8";
      ctx.fillText("分", centerX_Right + (isA5 ? 140 : 100), scoreY); ctx.textAlign = "left";
    }

    // --- Detailed Scores (Left Side) ---
    let barY = layout.barStartY; 
    const barMaxWidth = w - (isA5 ? 600 : 450); 
    
    (essay.detailedScores || []).slice(0, 4).forEach(ds => {
      // Label
      ctx.fillStyle = "#475569"; ctx.font = `bold ${layout.barLabelSize}px "Microsoft JhengHei", sans-serif`; 
      ctx.fillText(ds.label, layout.margin + 50, barY);
      
      const total = ds.total && ds.total > 0 ? ds.total : 10; 
      const score = ds.score || 0;
      
      // Bar
      const barH = isA5 ? 30 : 20; 
      const barX = layout.margin + 50; 
      const barY_Rect = barY + (isA5 ? 20 : 15);
      const barFullWidth = barMaxWidth - 100;

      // 1. Draw Background & Outline
      ctx.fillStyle = "#ffffff"; 
      ctx.fillRect(barX, barY_Rect, barFullWidth, barH);
      
      // Outline - Pure Black for B&W printing
      ctx.strokeStyle = "#000000"; 
      ctx.lineWidth = isA5 ? 4 : 3; // ~1pt visual weight
      ctx.strokeRect(barX, barY_Rect, barFullWidth, barH);

      // 2. Draw Fill
      ctx.fillStyle = "#6366f1"; 
      const ratio = total > 0 ? Math.min(1, Math.max(0, score / total)) : 0;
      const fillW = ratio * barFullWidth;
      ctx.fillRect(barX, barY_Rect, fillW, barH);
      
      // 3. Draw Score Text (Non-overlapping)
      if (mode === 'teacher') {
        ctx.fillStyle = "#6366f1"; 
        ctx.font = `bold ${layout.barLabelSize * 0.9}px "Microsoft JhengHei", sans-serif`; 
        ctx.textAlign = "left"; 
        ctx.fillText(`${score}/${total}`, barX + barFullWidth + (isA5 ? 20 : 15), barY + (isA5 ? 45 : 35)); 
      }
      barY += layout.barGap; 
    });

    let dynamicContentY = barY; 

    // --- Originality Section (Teacher Only, Bottom Left) ---
    if (mode === 'teacher' && essay.originality && checkOriginality) {
        const aiRatio = essay.originality.aiRatio || 0;
        const copyRatio = essay.originality.copyRatio || 0;
        const reason = essay.originality.reason || "無異常。";
        
        const riskY = barY; 
        
        ctx.font = `bold ${layout.barLabelSize}px "Microsoft JhengHei", sans-serif`; 
        ctx.fillStyle = "#475569";
        ctx.fillText("🔍 原創性檢核", layout.margin + 50, riskY);

        const getRiskColor = (ratio) => {
            if (ratio >= 40) return "#ef4444"; // Red (High)
            if (ratio >= 20) return "#f59e0b"; // Orange (Med)
            return "#22c55e"; // Green (Low)
        };

        // AI Risk Badge
        const aiColor = getRiskColor(aiRatio);
        const aiBadgeX = layout.margin + 50;
        const aiBadgeY = riskY + (isA5 ? 30 : 20);
        ctx.fillStyle = aiColor;
        ctx.beginPath(); ctx.roundRect(aiBadgeX, aiBadgeY, isA5 ? 300 : 220, isA5 ? 60 : 40, 10); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.font = `bold ${layout.barLabelSize * 0.8}px "Microsoft JhengHei", sans-serif`;
        ctx.fillText(`AI 佔比: ${aiRatio}%`, aiBadgeX + 20, aiBadgeY + (isA5 ? 42 : 28));

        // Copy Risk Badge
        const copyColor = getRiskColor(copyRatio);
        const copyBadgeX = aiBadgeX + (isA5 ? 320 : 240);
        ctx.fillStyle = copyColor;
        ctx.beginPath(); ctx.roundRect(copyBadgeX, aiBadgeY, isA5 ? 300 : 220, isA5 ? 60 : 40, 10); ctx.fill();
        ctx.fillStyle = "#fff"; 
        ctx.fillText(`抄襲佔比: ${copyRatio}%`, copyBadgeX + 20, aiBadgeY + (isA5 ? 42 : 28));

        // Reason - V15.0.0: Always show for clarity, adjust color based on risk
        const isRisky = aiRatio >= 20 || copyRatio >= 20;
        ctx.fillStyle = isRisky ? "#ef4444" : "#64748b"; 
        ctx.font = `bold ${layout.barLabelSize * 0.7}px "Microsoft JhengHei", sans-serif`;
        
        let displayReason = reason.substring(0, 45);
        if (reason.length > 45) displayReason += "...";
        
        ctx.fillText(`判定結果: ${displayReason}`, aiBadgeX, aiBadgeY + (isA5 ? 100 : 70));
        dynamicContentY = aiBadgeY + (isA5 ? 120 : 90); 
    }


    // --- Comments Section ---
    const commentsStartY = Math.max(layout.commentsStartY, dynamicContentY + (isA5 ? 60 : 40)); 

    const contentWidth = w - (layout.margin * 2) - 100;
    const maxYLimit = h - 150; 

    ctx.font = `bold ${layout.commentTitleSize}px "Microsoft JhengHei", sans-serif`; ctx.fillStyle = "#4f46e5"; 
    ctx.fillText("✨ 文章亮點", layout.margin + 50, commentsStartY);
    
    ctx.font = `${layout.commentBodySize}px "Microsoft JhengHei", sans-serif`; 
    
    // 確保最多只畫兩點重點
    const limitPoints = (txt) => {
        if (!txt) return "";
        const lines = txt.split('\n').filter(l => l.trim().length > 0);
        return lines.slice(0, 2).join('\n');
    };

    const rawHighlights = mode === 'student' ? stripCodes(essay.highlights) : essay.highlights;
    const rawSuggestions = mode === 'student' ? stripCodes(essay.suggestions) : essay.suggestions;
    
    const highlightsText = limitPoints(rawHighlights);
    const suggestionsText = limitPoints(rawSuggestions);

    let nextY = drawBulletText(ctx, highlightsText, layout.margin + 90, commentsStartY + (isA5 ? 80 : 60), contentWidth, layout.commentLineHeight, "#334155", maxYLimit, isA5 ? 1 : 0.8);

    // Suggestions Section
    nextY = Math.max(nextY + (isA5 ? 60 : 40), commentsStartY + (isA5 ? 300 : 220)); 

    if (nextY < maxYLimit - (isA5 ? 200 : 150)) { 
        ctx.font = `bold ${layout.commentTitleSize}px "Microsoft JhengHei", sans-serif`; ctx.fillStyle = "#f59e0b";
        ctx.fillText("💡 這樣寫會更好", layout.margin + 50, nextY);
        ctx.font = `${layout.commentBodySize}px "Microsoft JhengHei", sans-serif`; 
        nextY = drawBulletText(ctx, suggestionsText, layout.margin + 90, nextY + (isA5 ? 80 : 60), contentWidth, layout.commentLineHeight, "#334155", maxYLimit, isA5 ? 1 : 0.8);
    }

    // --- Footer ---
    const footerY = layout.footerY;

    if (mode === 'teacher') {
        const curriculumRegex = /(?:國語|國文|自然|社會|歷史|地理|公民|自|社|歷|地|公|tr|tm|pa|po|pc|pe)\s*[a-zA-Z0-9\-\u2160-\u216F\.]+/g;
        const allText = `${essay.highlights} ${essay.suggestions}`;
        const matches = [...new Set(allText.match(curriculumRegex) || [])]; 

        if (matches.length > 0) {
            ctx.font = `bold ${layout.footerSize}px "Microsoft JhengHei", sans-serif`; 
            ctx.fillStyle = "#64748b";
            ctx.fillText("📌 課綱對應：", layout.margin + 50, footerY);
            
            let tagX = layout.margin + (isA5 ? 290 : 220); let tagY = footerY - (isA5 ? 45 : 35);
            const paddingX = isA5 ? 20 : 15; const gapX = isA5 ? 15 : 10; const maxWidth = w - 400;
            ctx.font = `bold ${layout.footerSize * 0.9}px "Microsoft JhengHei", sans-serif`;

            matches.forEach(tag => {
                const textMetrics = ctx.measureText(tag);
                const tagWidth = textMetrics.width + paddingX * 2;
                if (tagX + tagWidth > maxWidth) return; 
                ctx.fillStyle = "#eff6ff"; ctx.strokeStyle = "#bfdbfe"; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.roundRect(tagX, tagY, tagWidth, isA5 ? 60 : 45, 15); ctx.fill(); ctx.stroke();
                ctx.fillStyle = "#2563eb"; ctx.fillText(tag, tagX + paddingX, tagY + (isA5 ? 40 : 30));
                tagX += tagWidth + gapX;
            });
        }
    }

    ctx.font = `italic ${layout.footerSize}px "Microsoft JhengHei", sans-serif`; ctx.fillStyle = "#94a3b8"; ctx.textAlign = "right";
    ctx.fillText(`日期：${gradingDate}`, w - layout.margin - 30, footerY);

    return canvas.toDataURL('image/png');
  };

  const handleSingleCardDownload = async (essay, mode) => {
    try {
        const dataUrl = await drawEssayCard(essay, mode);
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = `${essay.studentNumber || ''}_${essay.studentName || 'student'}_${mode === 'teacher' ? '教師版' : '學生版'}.png`;
        link.click();
    } catch (e) {
        console.error(e);
        showToast("圖片生成失敗", "error");
    }
  };

  const handleDownloadPDFReport = async (mode) => {
    if (essays.filter(e => e.status === 'completed').length === 0) return showToast("沒有已完成的批改紀錄", "error");
    
    setIsProcessing(true);
    showToast("正在產生 PDF 報告...");

    const isA5 = textTypeId === 'auto'; // Auto mode uses A5 cards, others use A6

    try {
      const jsPDF = await loadJsPDF();
      // Start with Landscape A4 if using A5 cards (2-up), Portrait if A6 (4-up)
      const orientation = isA5 ? 'landscape' : 'portrait';
      const doc = new jsPDF.jsPDF({ orientation: orientation, format: 'a4' });
      
      // V13.5.0: Enhanced Cut Lines Helper
      const drawCutLines = () => {
          const w = doc.internal.pageSize.getWidth();
          const h = doc.internal.pageSize.getHeight();
          
          doc.setDrawColor(150, 150, 150); // Darker Gray for visibility
          doc.setLineWidth(0.5);
          doc.setLineDash([5, 5], 0); // Dashed line
          
          // Draw lines on top of everything
          if (orientation === 'landscape') {
              // A5 2-up: Vertical line in middle
              doc.line(w/2, 0, w/2, h);
          } else {
              // A6 4-up: Vertical and Horizontal middle lines (Cross)
              doc.line(w/2, 0, w/2, h);
              doc.line(0, h/2, w, h/2);
          }
          doc.setLineDash([]); // Reset
          doc.setDrawColor(0); 
      };

      // 1. Teacher mode summary pages (Always Portrait)
      let hasAddedSummary = false;
      if (mode === 'teacher' && classAnalysis) {
        const summaryPages = generateSummaryPages(classAnalysis, gradeLevel, essayTopic, gradingDate, essays.filter(e => e.status === 'completed'));
        
        if (summaryPages.length > 0) {
            hasAddedSummary = true;
            summaryPages.forEach((pageImg, index) => {
                if (index === 0) {
                    doc.deletePage(1); // Remove default page
                    doc.addPage('a4', 'portrait');
                } else {
                    doc.addPage('a4', 'portrait');
                }
                doc.addImage(pageImg, 'PNG', 0, 0, 210, 297);
            });
        }
      }

      // 2. Add individual essay cards
      const completedEssays = essays.filter(e => e.status === 'completed');
      
      for (let i = 0; i < completedEssays.length; i++) {
        const essay = completedEssays[i];
        const cardDataUrl = await drawEssayCard(essay, mode);
        
        if (isA5) {
            // A5 Cards on A4 Landscape (2-up)
            const positionIndex = i % 2;
            const CARD_W = 148.5; const CARD_H = 210;
            
            if (positionIndex === 0) {
                // New page needed
                if (hasAddedSummary || i > 0) {
                    doc.addPage('a4', 'landscape');
                }
            }
            
            const x = positionIndex * CARD_W;
            doc.addImage(cardDataUrl, 'PNG', x, 0, CARD_W, CARD_H);

            // Draw cut lines after both cards are placed or if it's the last card
            if (positionIndex === 1 || i === completedEssays.length - 1) {
                drawCutLines();
            }
            
        } else {
            // A6 Cards on A4 Portrait (4-up)
            const positionIndex = i % 4;
            const CARD_W = 105; const CARD_H = 148.5;
            
            if (positionIndex === 0) {
                // New page needed
                if (hasAddedSummary || i > 0) {
                    doc.addPage('a4', 'portrait');
                }
            }
            
            const col = positionIndex % 2;
            const row = Math.floor(positionIndex / 2);
            const x = col * CARD_W;
            const y = row * CARD_H;
            doc.addImage(cardDataUrl, 'PNG', x, y, CARD_W, CARD_H);
            
            // Draw cut lines after last card of page or last card of all
            if (positionIndex === 3 || i === completedEssays.length - 1) {
                drawCutLines();
            }
        }
      }

      doc.save(`${essayTopic || '批改報告'}_${mode === 'teacher' ? '教師版' : '學生版'}.pdf`);
      showToast("PDF 下載成功");

    } catch (e) {
      console.error(e);
      showToast("PDF 生成失敗: " + e.message, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  const callGeminiWithRetry = async (modelId, payload, retries = 5) => {
    let delay = 1000;
    for (let i = 0; i < retries; i++) {
      if (stopRef.current) throw new Error("使用者停止"); 
      try {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${apiKey}`,
          { 
              method: 'POST', 
              headers: { 'Content-Type': 'application/json' }, 
              body: JSON.stringify(payload),
              signal: abortControllerRef.current?.signal 
          });
        if (response.ok) return await response.json();
        if (response.status !== 429 && response.status < 500) break;
      } catch (e) {
          if (e.name === 'AbortError' || stopRef.current) throw new Error("使用者停止");
      }
      await sleep(delay); delay *= 2;
    }
    throw new Error("API 調用失敗");
  };

  const processEssay = async (essayId) => {
    const essay = essaysRef.current.find(e => e.id === essayId);
    if (!essay) return false;
    setEssays(prev => prev.map(e => e.id === essayId ? { ...e, status: 'processing' } : e));
    
    try {
      // Step 1: Grading and OCR
      const imageParts = await Promise.all(essay.images.map(async (img) => {
        const base64 = await new Promise(resolve => {
          const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]); reader.readAsDataURL(img.file);
        });
        return { inlineData: { mimeType: img.file.type, data: base64 } };
      }));
      
      let systemPrompt = "";
      
      if (textTypeId === 'auto') {
        let rubricContent = "";
        if (gradeLevel.includes("國小")) rubricContent = RUBRIC_CONTENT_ELEMENTARY;
        else if (gradeLevel.includes("國中")) rubricContent = RUBRIC_CONTENT_JUNIOR;
        else rubricContent = RUBRIC_CONTENT_SENIOR;

        systemPrompt = `Role: 你是一位精通台灣 108 課綱（自然科學、社會領域、國語文）的資深教育評量專家。
        
        Task: 請針對使用者提供的「跨領域多元寫作文本」進行評改。
        
        # Step 1: Text Classification & Stage Detection
        1. **判斷領域**：
           - 若文本涉及實驗、變因、自然現象、模型建構 -> 分類為 **[自然科學探究]**。
           - 若文本涉及價值判斷、社會議題、歷史脈絡、人際互動 -> 分類為 **[社會領域探究]**。
        2. **判斷階段**：依據設定為 ${gradeLevel}。

        # Step 2: Apply 5-Level Rubric
        請嚴格依據以下定義的詳細 5 級分量表 (L1-L5) 進行評分：
        ${rubricContent}

        # Step 3: Output Format
        請依序輸出，並務必將等級 (L1-L5) 轉換為數值分數以供系統解析 (L5=100, L4=85, L3=75, L2=60, L1=40，可視情況微調)：
        
        1. **基本資訊**：判定領域、判定年級。
        2. **評分表** (請嚴格遵守以下格式，絕對不要使用 Markdown 表格，請使用單行條列)：
        ## 評分表
        [探究與邏輯]: 35/40
        [證據與資料]: 30/30
        [表達與結構]: 25/30

        3. **綜合評語**
        【重要比例原則】：60% 具體引用 108 課綱條目（如：(6-III-2)），40% 使用白話、常見文本審閱用詞（如：流暢、破題精彩等）。
        嚴禁直接複製等級描述。

        ## 文章亮點
        (條列式，最多 2 點，每點不超過 30 字，結合課綱與白話評語)

        ## 這樣寫會更好
        (條列式，最多 2 點，每點不超過 30 字，結合課綱與白話評語)
        
        ## 總分
        (上述三項分數加總數字)

        ## 文本類型判定
        [自然科學探究] 或 [社會領域探究]
        
        ## OCR 辨識出的原文
        (請一字不漏地轉錄圖片中的所有文字，這對後續處理很重要)`;

      } else if (textTypeId === 'Bf') {
        systemPrompt = `你是一位專業詩歌老師，熟悉台灣十二年國教國語文領域。本次批改「新詩/童詩」。級別：${gradeLevel}。
          
          【課綱與白話比例】
          請遵循『60/40 黃金比例』：約 60% 評語需具體引用並標註 108 課綱條目（如：(6-I-1)），其餘 40% 使用白話、常見文本審閱用詞。
          
          輸出格式 (請嚴格遵守，絕對不要使用 Markdown 表格)：
          ## 評分表
          [意象]: 10/30
          [形式]: 10/20
          [修辭]: 10/20
          [內容]: 10/30
          
          ## 文章亮點
          (條列式，最多 2 點，每點不超過 30 字，結合課綱與白話評語)
          
          ## 這樣寫會更好
          (條列式，最多 2 點，每點不超過 30 字，結合課綱與白話評語)
          
          ## 總分
          (數字)
          
          ## OCR 辨識出的原文
          (一字不漏轉錄)`;
      } else {
        const textType = TEXT_TYPES.find(t => t.id === textTypeId) || TEXT_TYPES[1];
        const curriculumContext = CURRICULUM_PERFORMANCE[gradeLevel] ? `\n\n參考課綱指標:\n${CURRICULUM_PERFORMANCE[gradeLevel].join('\n')}` : "";
        
        systemPrompt = `你是一位專業國語文寫作評量專家。題目：${essayTopic}。級別：${gradeLevel}。文體：${textType.label}。
          
          【評分規準】
          1. 立意與取材 (35-40%)
          2. 結構與組織 (25-30%)
          3. 遣詞與造句 (20-25%)
          4. 字體與格式 (10%)
          
          【課綱與白話比例】
          請遵循『60/40 黃金比例』：約 60% 評語需具體引用 108 課綱條目，其餘 40% 使用白話、常見文本審閱用詞（如：行雲流水、引人入勝等）。
          
          輸出格式 (請嚴格遵守，絕對不要使用 Markdown 表格)：
          ## 評分表
          [立意與取材]: 32/40
          [結構與組織]: 24/30
          [遣詞與造句]: 16/20
          [字體與格式]: 9/10
          
          ## 文章亮點
          (條列式，最多 2 點，每點不超過 30 字，結合課綱與白話評語)
          
          ## 這樣寫會更好
          (條列式，最多 2 點，每點不超過 30 字，結合課綱與白話評語)
          
          ## 總分
          (數字)
          
          ## OCR 辨識出的原文
          (一字不漏轉錄)
          ${curriculumContext}`;
      }

      const payload = { contents: [{ parts: [{ text: systemPrompt }, ...imageParts] }] };
      
      let gradingResult = null;
      let usedModelName = "";
      let usedModelId = "";

      for (const model of FALLBACK_MODEL_ORDER) {
        try {
          gradingResult = await callGeminiWithRetry(model.id, payload);
          usedModelName = model.name;
          usedModelId = model.id;
          break;
        } catch (e) {
          console.warn(`Model ${model.name} failed, trying next...`);
          if (stopRef.current) break; 
          continue;
        }
      }

      if (!gradingResult) throw new Error("所有 AI 模型皆無法回應，請稍後再試。");

      const text = gradingResult.candidates?.[0]?.content?.parts?.[0]?.text || "辨識失敗。";
      
      // Parse Grading Logic
      const typeMatch = text.match(/## 文本類型判定\s*[:：]?\s*\n?\s*\[?(.*?)\]?/);
      const evaluatedType = typeMatch ? typeMatch[1].replace(/[\[\]]/g, '') : null;
      
      // Extract OCR strictly
      let ocrText = "無法獲取原文內容";
      if (text.includes("## OCR 辨識出的原文")) {
          ocrText = text.split("## OCR 辨識出的原文")[1]?.trim();
      }
      
      const scoreMatch = text.match(/## 總分\s*[:：]?\s*(\d+)/);
      
      let highlights = "";
      if (text.includes("## 文章亮點")) highlights = text.split("## 文章亮點")[1]?.split("##")[0]?.replace(/\*/g, '').trim();
      
      let suggestions = "";
      if (text.includes("## 這樣寫會更好")) suggestions = text.split("## 這樣寫會更好")[1]?.split("##")[0]?.replace(/\*/g, '').trim();

      const scores = [];
      const tableMatch = text.match(/#*\s*評分表([\s\S]*?)(?=\n##|$)/);
      const tableContent = tableMatch ? tableMatch[1] : text;

      if (tableContent) {
          const lines = tableContent.split('\n');
          for (const line of lines) {
              const cleanLine = line.replace(/\*/g, '').trim();
              if (!cleanLine) continue;
              
              if (cleanLine.includes('|')) {
                  const parts = cleanLine.split('|').map(p => p.trim()).filter(p => p);
                  if (parts.length >= 2 && !cleanLine.includes('---')) {
                      const label = parts[0].replace(/[\*\[\]\-\d\.]/g, '').trim();
                      const scoreStr = parts[parts.length - 1];
                      const smatch = scoreStr.match(/(\d+)(?:\s*\/\s*(\d+))?/);
                      if (smatch && label.length >= 2 && !label.includes("項目") && !label.includes("分數") && !label.includes("總分")) {
                          let total = smatch[2] ? parseInt(smatch[2]) : 0;
                          if (total === 0 && textTypeId === 'auto') total = scores.length === 0 ? 40 : 30;
                          scores.push({ label, score: parseInt(smatch[1]), total });
                      }
                  }
                  continue;
              }

              const match = cleanLine.match(/^(?:[\-\d\.\s\[【]*)([^\]】:：\n]+)(?:[\]】\s]*)[::：]\s*(\d+)(?:\s*\/\s*(\d+))?/);
              
              if (match) {
                  const label = match[1].trim();
                  if (label.length < 2 || label.includes("評分") || label.includes("總分") || label.toLowerCase().includes("total")) continue;
                  
                  const score = parseInt(match[2]);
                  let total = match[3] ? parseInt(match[3]) : 0;
                  
                  if (total === 0 && textTypeId === 'auto') {
                      total = scores.length === 0 ? 40 : 30;
                  }
                  scores.push({ label, score, total });
              }
          }
      }

      if (scores.length === 0) throw new Error("AI 回傳格式不符，無法讀取分數與評語 (請重試)");

      let finalScore = scoreMatch ? scoreMatch[1] : "N/A";
      if ((finalScore === "N/A" || isNaN(parseInt(finalScore))) && scores.length > 0) {
          finalScore = scores.reduce((acc, curr) => acc + curr.score, 0).toString();
      }

      // Step 2: Originality Check (Independent Call)
      let originalityData = null;
      if (checkOriginality && ocrText && ocrText !== "無法獲取原文內容") {
          try {
              const origPayload = {
                  contents: [{ parts: [{ text: ocrText }] }],
                  systemInstruction: { parts: [{ text: ORIGINALITY_SYSTEM_PROMPT }] },
                  // Use standard prompt to ensure we get a JSON back
                  generationConfig: { responseMimeType: "application/json" }
              };
              
              const origResult = await callGeminiWithRetry(usedModelId, origPayload);
              const rawJson = origResult.candidates?.[0]?.content?.parts?.[0]?.text;
              
              if (rawJson) {
                  const jsonMatch = rawJson.match(/\{[\s\S]*\}/);
                  const cleanedJson = jsonMatch ? jsonMatch[0] : rawJson.replace(/```json/gi, '').replace(/```/g, '').trim();
                  const parsedOrig = JSON.parse(cleanedJson);
                  const segments = parsedOrig.segments || [];
                  const totalSegs = segments.length;
                  
                  if (totalSegs > 0) {
                      const aiCount = segments.filter(s => s.type === 'ai').length;
                      const copyCount = segments.filter(s => s.type === 'plagiarism').length;
                      
                      const aiRatio = Math.round((aiCount / totalSegs) * 100);
                      const copyRatio = Math.round((copyCount / totalSegs) * 100);
                      const originalRatio = Math.max(0, 100 - aiRatio - copyRatio);
                      
                      // Extract top reasons for flagged content
                      const reasons = segments
                          .filter(s => s.type === 'ai' || s.type === 'plagiarism')
                          .map(s => s.reason)
                          .filter(r => r) // remove empty
                          .slice(0, 2) // Limit to top 2 to keep UI clean
                          .join('；');

                      originalityData = {
                          aiRatio,
                          copyRatio,
                          originalRatio,
                          reason: reasons || "文本判定為人類原創，無明顯異常特徵。",
                          segments
                      };
                  }
              }
          } catch (origError) {
              console.error("Originality check failed:", origError);
              originalityData = {
                  aiRatio: 0, copyRatio: 0, originalRatio: 100,
                  reason: "原創性分析服務暫時無法回應，請稍後重試。"
              };
          }
      }

      setEssays(prev => prev.map(e => e.id === essayId ? { 
        ...e, 
        status: 'completed', 
        ocrText: ocrText, 
        score: finalScore, 
        highlights, 
        suggestions, 
        detailedScores: scores,
        originality: originalityData,
        usedModel: usedModelName,
        evaluatedType // Store type only if found (Auto mode)
      } : e));
      return true;
    } catch (err) { 
        if (stopRef.current || err.message === "使用者停止") {
             setEssays(prev => prev.map(e => e.id === essayId ? { ...e, status: 'pending' } : e));
             return false;
        }
        setEssays(prev => prev.map(e => e.id === essayId ? { ...e, status: 'error', response: err.message } : e)); 
        return false; 
    }
  };

  const startBatchInternal = async (targets) => {
      if (!apiKey) return showToast("請輸入 API Key", "error");
      setIsProcessing(true); 
      stopRef.current = false;
      abortControllerRef.current = new AbortController();
      
      let successCount = 0;
      let failCount = 0;

      for (let i = 0; i < targets.length; i++) {
        if (stopRef.current) break;
        const success = await processEssay(targets[i].id); 
        if (success) successCount++; else if (!stopRef.current) failCount++;
        setCurrentProgress(((i + 1) / targets.length) * 100);
      }
      setIsProcessing(false); 
      
      if (!stopRef.current) {
          setTimeout(() => {
              // Calculate stats from the latest essaysRef based on new percentage logic
              let h = 0, m = 0, l = 0;
              if (checkOriginality) {
                  const targetIds = new Set(targets.map(t => t.id));
                  essaysRef.current.forEach(e => {
                      if (targetIds.has(e.id) && e.status === 'completed' && e.originality) {
                           const maxRisk = Math.max(e.originality.aiRatio, e.originality.copyRatio);
                           if (maxRisk >= 40) h++;
                           else if (maxRisk >= 20) m++;
                           else l++;
                      }
                  });
              }

              generateSummary();
              setBatchResult({ 
                  total: targets.length, 
                  success: successCount, 
                  fail: failCount,
                  originalityStats: checkOriginality ? { high: h, medium: m, low: l } : null
              });
          }, 500);
      }
  };

  const startBatch = async () => {
    const pending = essaysRef.current.filter(e => e.status !== 'completed');
    await startBatchInternal(pending);
  };

  const handleRegradeClick = () => { if (essays.length > 0) setShowRegradeConfirm(true); };

  const performRegradeAll = async () => {
      setShowRegradeConfirm(false); 
      const resetList = essays.map(e => ({ ...e, status: 'pending', score: 'N/A', ocrText: '', highlights: '', suggestions: '', detailedScores: [], response: null, usedModel: null, originality: null }));
      setEssays(resetList); essaysRef.current = resetList;
      await startBatchInternal(resetList);
  };

  const handleClearClick = () => { if (essays.length > 0) setShowClearConfirm(true); };

  const performClearAll = () => {
      setEssays([]); setBatchResult(null); setClassAnalysis(''); setShowClearConfirm(false);
      clearDB().then(() => showToast("已清空所有資料"));
  };

  const generateSummary = async () => {
    const completed = essaysRef.current.filter(e => e.status === 'completed');
    if (completed.length === 0) return;
    setIsGeneratingSummary(true); 
    try {
      const res = await callGeminiWithRetry(FALLBACK_MODEL_ORDER[0].id, { contents: [{ parts: [{ text: `請總結全班作文表現。份數：${completed.length}。題目：${essayTopic}` }] }] });
      setClassAnalysis(res.candidates?.[0]?.content?.parts?.[0]?.text || "");
    } catch (e) { console.error(e); } finally { setIsGeneratingSummary(false); }
  };

  // --- File Conversion Logic (From App 1) ---
  const convertPdfToImages = async (file) => {
    await loadScript("https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js");
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    const images = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 2.0 });
      const canvas = document.createElement('canvas'); const context = canvas.getContext('2d');
      canvas.height = viewport.height; canvas.width = viewport.width;
      await page.render({ canvasContext: context, viewport: viewport }).promise;
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      images.push(new File([blob], `${file.name}-page-${i}.png`, { type: 'image/png' }));
    }
    return images;
  };

  const convertDocxToImages = async (file) => {
    await loadScript("https://cdnjs.cloudflare.com/ajax/libs/mammoth/1.6.0/mammoth.browser.min.js");
    const arrayBuffer = await file.arrayBuffer();
    const result = await window.mammoth.extractRawText({ arrayBuffer: arrayBuffer });
    const imageFile = await renderTextToImage(result.value, file.name);
    return [imageFile];
  };

  const convertPptxToImages = async (file) => {
    await loadScript("https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js");
    const zip = await window.JSZip.loadAsync(file);
    let fullText = "";
    const slideFiles = Object.keys(zip.files).filter(name => name.startsWith("ppt/slides/slide") && name.endsWith(".xml"));
    slideFiles.sort((a, b) => parseInt(a.match(/slide(\d+)\.xml/)[1]) - parseInt(b.match(/slide(\d+)\.xml/)[1]));
    for (const slidePath of slideFiles) {
        const slideXml = await zip.file(slidePath).async("string");
        const parser = new DOMParser(); const xmlDoc = parser.parseFromString(slideXml, "text/xml");
        const texts = xmlDoc.getElementsByTagName("a:t");
        for (let i = 0; i < texts.length; i++) { fullText += texts[i].textContent + "\n"; }
        fullText += "\n---\n";
    }
    if (!fullText.trim()) fullText = "(無法提取文字或內容為純圖片)";
    const imageFile = await renderTextToImage(fullText, file.name);
    return [imageFile];
  };

  const handleFiles = async (files) => {
    if (!files || files.length === 0) return;
    setIsLoadingFile(true);
    try {
      const filesArray = Array.from(files);
      const processedGroups = []; 

      for (const file of filesArray) {
        let processedImages = [];
        if (file.type === "application/pdf") {
          showToast(`正在解析 PDF: ${file.name}...`);
          processedImages = await convertPdfToImages(file);
        } else if (file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
          showToast(`正在解析 Word: ${file.name}...`);
          processedImages = await convertDocxToImages(file);
        } else if (file.type === "application/vnd.openxmlformats-officedocument.presentationml.presentation") {
          showToast(`正在解析 PowerPoint: ${file.name}...`);
          processedImages = await convertPptxToImages(file);
        } else if (file.type.startsWith("image/")) {
          processedImages = [file];
        } else {
          showToast(`不支援的檔案格式: ${file.name}`, 'error');
          continue;
        }

        const nameWithoutExt = file.name.split('.').slice(0, -1).join('.');
        let groupingId = nameWithoutExt;
        let pageOffset = 0;

        const pageMatch = nameWithoutExt.match(/^(.*)[-_](\d+)$/);
        if (pageMatch) {
            groupingId = pageMatch[1];
            pageOffset = parseInt(pageMatch[2], 10);
        }

        let group = processedGroups.find(g => g.id === groupingId);
        if (!group) {
            group = { id: groupingId, images: [] };
            processedGroups.push(group);
        }

        processedImages.forEach((imgFile, idx) => {
             group.images.push({ 
                 file: imgFile, 
                 sortKey: pageOffset * 1000 + idx 
             });
        });
      }

      const newEssays = processedGroups.map(group => {
         group.images.sort((a, b) => a.sortKey - b.sortKey);
         const finalImages = group.images.map(wrapper => ({
             file: wrapper.file,
             preview: URL.createObjectURL(wrapper.file)
         }));

         let rawName = group.id;
         let studentName = rawName;
         let studentNumber = "";
         
         const prefixMatch = rawName.match(/^(\d+)[\s_\-.]+(.*)$/);
         if (prefixMatch) {
            studentNumber = prefixMatch[1]; 
            studentName = prefixMatch[2]; 
         } else {
            const suffixMatch = rawName.match(/^(.*?)[\s_\-.]?(\d+)$/);
            if (suffixMatch) {
                studentName = suffixMatch[1];
                studentNumber = suffixMatch[2];
            }
         }
         
         return {
            id: Math.random().toString(36).substr(2, 9),
            studentName: studentName.trim() || "未命名",
            studentNumber: studentNumber,
            status: 'pending',
            images: finalImages,
            score: "N/A"
         };
      });

      setEssays(prev => [...prev, ...newEssays]);
      showToast(`成功載入 ${newEssays.length} 份作業`);

    } catch (e) { console.error(e); showToast("檔案解析失敗: " + e.message, 'error'); } 
    finally { setIsLoadingFile(false); }
  };

  return (
    <div className="min-h-screen bg-[#F0F4F8] text-slate-800 pb-20 font-sans selection:bg-indigo-200 selection:text-indigo-900">
      <nav className="bg-white/80 backdrop-blur-xl border-b border-slate-200 sticky top-0 z-40 px-6 h-16 flex items-center justify-between shadow-sm">
        <div className="flex items-center space-x-2 text-indigo-700">
          <FileSpreadsheet className="bg-indigo-600 p-1.5 rounded-lg text-white" />
          <h1 className="text-xl font-black">文本批改AI助手 <span className="text-slate-300 font-light">V15.0</span></h1>
        </div>
        <div className="flex gap-2">
            <button onClick={handleExportCSV} className="p-2 text-slate-400 hover:text-emerald-600 transition-colors" title="匯出成績單 (CSV)"><FileSpreadsheet size={20} /></button>
            <button onClick={() => restoreInputRef.current.click()} className="p-2 text-slate-400 hover:text-indigo-600 transition-colors" title="匯入紀錄"><UploadCloud size={20} /></button>
            <button onClick={handleExportBackup} className="p-2 text-slate-400 hover:text-indigo-600 transition-colors" title="備份紀錄"><DownloadCloud size={20} /></button>
            {essays.some(e => e.status === 'completed') && (
            <div className="flex gap-1 border-r pr-2 mr-2 border-slate-200">
              <button onClick={() => handleDownloadPDFReport('teacher')} className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-xl transition-all" title="教師版 PDF"><FileBadge size={20} /></button>
              <button onClick={() => handleDownloadPDFReport('student')} className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-xl transition-all" title="學生版 PDF"><GraduationCap size={20} /></button>
            </div>
            )}
            <button onClick={handleClearClick} className="p-2 text-slate-400 hover:text-rose-500 transition-colors"><Eraser size={20} /></button>
            <button onClick={() => setShowChangelog(true)} className="p-2 text-slate-400 hover:text-indigo-600 transition-colors"><History size={20} /></button>
        </div>
        <input type="file" ref={restoreInputRef} onChange={handleImportBackup} accept=".json" className="hidden" />
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-8 flex flex-col lg:flex-row gap-8">
        {/* Left Sidebar (Controls) */}
        <aside className="w-full lg:w-80 shrink-0 space-y-6">
          <div className="bg-white p-7 rounded-[2rem] border-0 shadow-[0_8px_30px_rgb(0,0,0,0.08)] space-y-7 lg:sticky lg:top-24 h-fit max-h-[calc(100vh-8rem)] overflow-y-auto custom-scrollbar pr-3">
            
            <div className="pb-4 border-b border-slate-100">
              <h3 className="text-lg font-black text-slate-800 mb-1 flex items-center gap-2"><Settings size={18} className="text-indigo-500"/> 設定與控制</h3>
              <p className="text-xs text-slate-400">請設定批改所需的各項參數</p>
            </div>

            <div>
              <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">Gemini API Key</label>
              <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} className="w-full px-4 py-3 bg-slate-50 border rounded-2xl outline-none" placeholder="Enter API Key" />
            </div>

            <div>
              <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">自訂獎牌圖示</label>
              <MedalSettings medals={customMedals} onUpload={handleMedalUpload} onReset={() => setCustomMedals({})} />
            </div>

            <div className="space-y-4">
              {/* Originality Toggle */}
              <div className="flex items-center justify-between p-3 bg-white rounded-xl border border-slate-200">
                  <div className="flex items-center gap-2">
                    <ShieldAlert size={16} className={checkOriginality ? "text-indigo-500" : "text-slate-400"} />
                    <span className="text-sm font-bold text-slate-700">啟用原創性檢測</span>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" checked={checkOriginality} onChange={e => setCheckOriginality(e.target.checked)} className="sr-only peer" />
                    <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                  </label>
              </div>

              <div>
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">年級標準</label>
                <select value={gradeLevel} onChange={e => setGradeLevel(e.target.value)} className="w-full p-3 bg-slate-50 border rounded-2xl text-sm outline-none">
                  {GRADE_LEVELS.map(g => <option key={g} value={g}>{g}</option>)}
                </select>
              </div>

              <div>
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">批改模式</label>
                <select value={textTypeId} onChange={e => setTextTypeId(e.target.value)} className="w-full p-3 bg-slate-50 border rounded-2xl text-sm outline-none">
                  {TEXT_TYPES.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
                </select>
              </div>

              <div>
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">作文題目</label>
                <input type="text" value={essayTopic} onChange={e => setEssayTopic(e.target.value)} className="w-full px-4 py-3 bg-slate-50 border rounded-2xl outline-none" placeholder="例如：我的夢想" />
              </div>

              <div>
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">批改日期</label>
                <input type="date" value={gradingDate} onChange={e => setGradingDate(e.target.value)} className="w-full p-3 bg-slate-50 border rounded-2xl outline-none text-sm" />
              </div>

              <div className="space-y-2">
                {isProcessing ? (
                  <button onClick={() => { stopRef.current = true; if (abortControllerRef.current) abortControllerRef.current.abort(); }} className="w-full bg-rose-500 text-white font-black text-lg py-4 rounded-2xl shadow-xl hover:bg-rose-600 flex items-center justify-center gap-2 transition-all active:scale-95">
                    <StopCircle size={24} /> <span>停止批改</span>
                  </button>
                ) : (
                  <>
                    <button onClick={startBatch} disabled={essays.length === 0} className={`w-full bg-gradient-to-r from-indigo-600 to-violet-600 text-white font-black text-lg py-4 rounded-2xl shadow-xl hover:shadow-2xl hover:scale-[1.02] disabled:from-slate-300 disabled:to-slate-400 disabled:shadow-none disabled:hover:scale-100 flex items-center justify-center gap-2 transition-all active:scale-95 ${essays.filter(e => e.status !== 'completed').length > 0 ? 'animate-pulse' : ''}`}>
                      <PlayCircle size={24} /> <span>啟動全體批改</span>
                    </button>
                    <button onClick={handleRegradeClick} disabled={essays.length === 0} className="w-full bg-orange-50 text-orange-600 font-bold py-3 rounded-xl border border-orange-200 hover:bg-orange-100 transition-all text-sm flex items-center justify-center gap-2">
                      <RotateCw size={16} /> <span>全部重新批改</span>
                    </button>
                  </>
                )}
              </div>

              <div className="space-y-3">
                <div className="text-center py-1">
                  <span className="text-[10px] font-bold text-slate-400 bg-slate-100 px-3 py-1 rounded-full">評分參考架構</span>
                </div>

                {textTypeId === 'auto' ? (
                    // Auto / Inquiry Mode Display
                    <>
                      <div className="p-4 bg-indigo-50/50 rounded-2xl border border-indigo-100">
                          <h4 className="text-xs font-black text-indigo-700 mb-3 flex items-center gap-1"><Microscope size={14} /> Route A: 自然科學探究</h4>
                          <div className="space-y-3">
                          {INQUIRY_RUBRICS_DISPLAY["Route A: 自然科學探究"].map((r, i) => (
                             <div key={i} className="text-[10px] leading-relaxed">
                             <span className="font-bold text-indigo-900">【{r.label}】({r.weight})</span>
                             <span className="text-slate-600 block mt-0.5">{r.desc}</span>
                             </div>
                          ))}
                          </div>
                    </div>
                    <div className="p-4 bg-orange-50/50 rounded-2xl border border-orange-100">
                        <h4 className="text-xs font-black text-orange-700 mb-3 flex items-center gap-1"><Landmark size={14} /> Route B: 社會領域探究</h4>
                        <div className="space-y-3">
                        {INQUIRY_RUBRICS_DISPLAY["Route B: 社會領域探究"].map((r, i) => (
                            <div key={i} className="text-[10px] leading-relaxed">
                            <span className="font-bold text-orange-900">【{r.label}】({r.weight})</span>
                            <span className="text-slate-600 block mt-0.5">{r.desc}</span>
                            </div>
                        ))}
                        </div>
                    </div>
                   </>
                ) : textTypeId === 'Bf' ? (
                    // Poetry Mode Display
                    <>
                      <div className="p-4 bg-indigo-50/50 rounded-2xl border border-indigo-100">
                        <h4 className="text-xs font-black text-indigo-700 mb-3 flex items-center gap-1"><Wind size={14} /> 新詩發展進程</h4>
                        <p className="text-[10px] leading-relaxed text-indigo-900 font-bold bg-white/50 p-2 rounded-lg border border-indigo-50">
                          {POETRY_PROGRESS[gradeLevel]}
                        </p>
                    </div>
                    <div className="p-4 bg-purple-50/50 rounded-2xl border border-purple-100">
                        <h4 className="text-xs font-black text-purple-700 mb-3 flex items-center gap-1"><Layers size={14} /> 詩歌評分指標</h4>
                        <div className="space-y-3">
                        {POETRY_RUBRICS.map((r, i) => (
                            <div key={i} className="text-[10px] leading-relaxed">
                            <span className="font-bold text-purple-900">【{r.label}】({r.weight})</span>
                            <span className="text-slate-600 block mt-0.5">{r.desc}</span>
                            </div>
                        ))}
                        </div>
                    </div>
                   </>
                ) : (
                    // Standard Composition Mode Display
                    <div className="p-4 bg-indigo-50/50 rounded-2xl border border-indigo-100">
                        <h4 className="text-xs font-black text-indigo-700 mb-3 flex items-center gap-1"><BookOpen size={14} /> 國語文評分規準</h4>
                        <div className="space-y-3">
                        {(STANDARD_SCORING_RUBRICS[gradeLevel] || []).map((r, i) => (
                            <div key={i} className="text-[10px] leading-relaxed">
                            <span className="font-bold text-indigo-900">【{r.label}】({r.weight})</span>
                            <span className="text-slate-600 block mt-0.5">{r.desc}</span>
                            </div>
                        ))}
                        </div>
                    </div>
                )}

                <div className="p-4 bg-rose-50/50 rounded-2xl border border-rose-100">
                  <h4 className="text-xs font-black text-rose-700 mb-3 flex items-center gap-1"><Medal size={14} /> 分數等第標準</h4>
                  <div className="grid grid-cols-2 gap-y-2 text-[10px] font-bold">
                    <div className="text-purple-600 flex items-center gap-1">特優: 93-100</div>
                    <div className="text-rose-600 flex items-center gap-1">優等: 85-92</div>
                    <div className="text-amber-500 flex items-center gap-1">金獎: 80-84</div>
                    <div className="text-slate-500 flex items-center gap-1">銀獎: 75-79</div>
                    <div className="text-orange-600 flex items-center gap-1">佳作: 70-74</div>
                    <div className="text-blue-600 flex items-center gap-1">普獎: 70以下</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </aside>

        <div className="flex-1 space-y-8 min-w-0">
          {isGeneratingSummary && !classAnalysis && (
             <div className="bg-white p-8 rounded-[2rem] border-0 shadow-[0_8px_30px_rgb(0,0,0,0.08)] flex items-center justify-center gap-3 text-indigo-600 animate-pulse">
                <Sparkles size={24} className="text-indigo-500" />
                <span className="font-bold text-lg">✨ 正在產生全班教學總結分析...</span>
             </div>
          )}

          {classAnalysis && (
            <div className="bg-white p-8 rounded-[2rem] border-0 shadow-[0_8px_30px_rgb(0,0,0,0.08)] animate-in slide-in-from-top duration-700 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500"></div>
              <div className="flex items-center gap-3 mb-6 mt-2">
                <School className="p-3 bg-indigo-600 rounded-xl text-white shadow-lg" size={48} />
                <h2 className="text-xl font-black text-indigo-900">全班教學總結分析</h2>
              </div>
              <SimpleMarkdown text={classAnalysis} />
              <ScoreDistributionChart essays={essays} />
            </div>
          )}

          <div 
            onClick={() => !isLoadingFile && fileInputRef.current.click()}
            onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={e => { e.preventDefault(); setIsDragging(false); handleFiles(e.dataTransfer.files); }}
            className={`group border-2 border-dashed p-12 rounded-[2rem] text-center cursor-pointer transition-all duration-300 relative ${isDragging ? 'border-indigo-500 bg-indigo-50 scale-[0.99] shadow-inner' : 'border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-white hover:shadow-xl hover:-translate-y-1'}`}
          >
            {isLoadingFile ? (
               <div className="flex flex-col items-center justify-center gap-4">
                  <Loader2 className="animate-spin text-indigo-500" size={48} />
                  <p className="font-bold text-slate-500">正在解析文件...</p>
               </div>
            ) : (
                <>
                    <div className="flex justify-center items-center gap-4 mb-4">
                        <Upload className="text-indigo-500" size={48} />
                        <div className="md:hidden" onClick={(e) => { e.stopPropagation(); cameraInputRef.current.click(); }}>
                            <div className="bg-indigo-100 p-3 rounded-full text-indigo-600 hover:bg-indigo-200 transition-colors shadow-md">
                                <Camera size={32} />
                            </div>
                        </div>
                    </div>
                    
                    <p className="font-bold text-slate-700">點擊或拖曳上傳 (圖片 / PDF / Word / PPT)</p>
                    <p className="text-xs text-slate-400 mt-1 italic">
                        支援格式: .jpg, .png, .pdf, .docx, .pptx <br/>
                        檔名規則: 01_姓名.pdf (自動解析座號姓名)
                    </p>
                </>
            )}
            
            <input type="file" ref={fileInputRef} onChange={e => handleFiles(e.target.files)} multiple className="hidden" accept="image/*,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation" />
            <input type="file" ref={cameraInputRef} capture="environment" accept="image/*" className="hidden" onChange={e => handleFiles(e.target.files)} />
          </div>

          <div className="space-y-6">
            {essays.map(essay => {
              const info = getGradeInfo(essay.score);
              return (
                <div key={essay.id} className="bg-white p-7 rounded-[2rem] border-0 shadow-[0_8px_30px_rgb(0,0,0,0.08)] flex flex-col md:flex-row gap-8 hover:shadow-[0_15px_40px_rgb(0,0,0,0.12)] hover:-translate-y-1 transition-all duration-300 relative overflow-hidden">
                  <div className={`absolute left-0 top-0 bottom-0 w-2 ${info.color.replace('text-', 'bg-')} opacity-80`}></div>
                  <div className="md:w-40 flex flex-col gap-2 shrink-0">
                    <div className="aspect-[3/4] bg-slate-100 rounded-xl overflow-hidden border relative group cursor-pointer" onClick={() => setViewingImage(essay.images[0].preview)}>
                      <img src={essay.images[0].preview} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" alt="Essay" />
                      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                          <Maximize2 className="text-white opacity-0 group-hover:opacity-100 drop-shadow-md" size={24} />
                      </div>
                      <div className="absolute bottom-1 right-1 bg-black/50 text-white text-[8px] px-1 rounded">P.1</div>
                      <button onClick={(e) => { e.stopPropagation(); setEssays(prev => prev.filter(e => e.id !== essay.id)); }} className="absolute top-2 right-2 p-1.5 bg-rose-500 text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-rose-600"><Trash2 size={14}/></button>
                    </div>
                    {essay.images.length > 1 && (
                      <div className="text-[10px] font-bold text-slate-400 text-center">共 {essay.images.length} 頁</div>
                    )}
                  </div>
                  <div className="flex-1 space-y-4">
                    <div className="flex justify-between items-start">
                      <div className="flex flex-wrap gap-3">
                        <div className="flex items-center space-x-2 bg-white px-3 py-1.5 rounded-xl border border-slate-200 shadow-sm">
                            <User size={14} className="text-slate-400" />
                            <input value={essay.studentName} onChange={(e) => setEssays(prev => prev.map(ev => ev.id === essay.id ? { ...ev, studentName: e.target.value } : ev))} className="font-bold outline-none bg-transparent w-20 text-slate-700 text-sm" placeholder="姓名" />
                        </div>
                        <div className="flex items-center space-x-2 bg-white px-3 py-1.5 rounded-xl border border-slate-200 shadow-sm">
                            <Hash size={14} className="text-slate-400" />
                            <input value={essay.studentNumber} onChange={(e) => setEssays(prev => prev.map(ev => ev.id === essay.id ? { ...ev, studentNumber: e.target.value } : ev))} className="font-bold outline-none bg-transparent w-12 text-slate-700 text-sm" placeholder="座號" />
                        </div>
                        <div className={`text-sm font-black flex items-center px-2 py-1 rounded-lg ${info.color} bg-opacity-10`}>{info.label}</div>
                        {essay.evaluatedType && (
                            <div className="text-[10px] font-bold bg-slate-100 text-slate-600 px-2 py-1 rounded-lg border border-slate-200 flex items-center gap-1">
                                {essay.evaluatedType.includes("自然") ? <Microscope size={12}/> : <Landmark size={12}/>}
                                {essay.evaluatedType.includes("自然") ? "自然科學類" : "社會領域類"}
                            </div>
                        )}
                        {/* Originality Risk Badges (Preview) */}
                        {essay.originality && (essay.originality.aiRatio >= 20) && (
                             <div className={`flex items-center space-x-1 px-2 py-1 rounded-lg text-xs font-bold border ${essay.originality.aiRatio >= 40 ? 'bg-red-100 text-red-600 border-red-200' : 'bg-orange-100 text-orange-600 border-orange-200'}`} title={`AI 比例: ${essay.originality.aiRatio}%`}>
                                 <ShieldAlert size={12} />
                                 <span>AI: {essay.originality.aiRatio}%</span>
                             </div>
                        )}
                      </div>
                      <div className="text-3xl font-black text-indigo-600 flex flex-col items-end">
                        <span>{essay.score}<span className="text-xs text-slate-400 ml-1">分</span></span>
                        {essay.usedModel && (
                          <span className="mt-1 text-[9px] text-slate-400 flex items-center gap-1 bg-slate-50 px-2 py-0.5 rounded-full border border-slate-100" title="批改模型">
                            <Cpu size={10} /> {essay.usedModel}
                          </span>
                        )}
                      </div>
                    </div>

                    {essay.status === 'completed' && (
                      <div className="flex items-center gap-2 flex-wrap py-3 border-y border-slate-100 my-4">
                        <button onClick={() => handleSingleCardDownload(essay, 'teacher')} className="flex items-center gap-1 text-[11px] font-bold bg-slate-100 text-slate-600 px-4 py-2 rounded-xl hover:bg-indigo-600 hover:text-white transition-all"><FileDown size={14} /> 教師版</button>
                        <button onClick={() => handleSingleCardDownload(essay, 'student')} className="flex items-center gap-1 text-[11px] font-bold bg-indigo-50 text-indigo-600 px-4 py-2 rounded-xl hover:bg-indigo-600 hover:text-white transition-all"><FileDown size={14} /> 學生版</button>
                        <button onClick={() => processEssay(essay.id)} className="flex items-center gap-1 text-[11px] font-bold bg-amber-50 text-amber-600 px-4 py-2 rounded-xl hover:bg-amber-500 hover:text-white transition-all border border-amber-100 ml-auto shadow-sm"><RotateCw size={14} /> 重新批改</button>
                      </div>
                    )}

                    {essay.status === 'completed' ? (
                      <div className="space-y-6 mt-2">
                        <div className="grid grid-cols-2 gap-x-8 gap-y-4">
                          {essay.detailedScores?.map((ds, idx) => (
                            <div key={idx} className="space-y-1.5 group/slider">
                              <div className="flex justify-between text-xs font-bold text-slate-500 uppercase">
                                <span>{ds.label}</span>
                                <span className="text-indigo-600">{ds.score}/{ds.total > 0 ? ds.total : 10}</span>
                              </div>
                              <div className="relative h-3 w-full rounded-full bg-slate-50 border border-slate-400 cursor-pointer shadow-sm">
                                <div className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-400 to-indigo-600 rounded-full transition-all duration-300" style={{ width: `${Math.min(100, ds.total > 0 ? (ds.score / ds.total) * 100 : 0)}%` }}>
                                    <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1.5 w-4 h-4 bg-white border-2 border-indigo-600 rounded-full shadow-sm z-10" />
                                </div>
                                <input type="range" min="0" max={ds.total > 0 ? ds.total : 10} value={ds.score} onChange={(e) => handleScoreChange(essay.id, idx, e.target.value)} className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer z-20" title={`調整分數: ${ds.score}`} />
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Originality Info Display in Expanded View - Percentage Based */}
                        {essay.originality && (
                             <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100 flex flex-col gap-3">
                                 <div className="flex items-center gap-2 mb-1">
                                     <ShieldAlert size={16} className="text-slate-400" />
                                     <span className="font-bold text-sm text-slate-700">原創性結構分析</span>
                                 </div>
                                 
                                 {/* Progress Bar Container */}
                                 <div className="h-4 w-full flex rounded-full overflow-hidden border border-slate-200">
                                     {essay.originality.aiRatio > 0 && <div style={{width: `${essay.originality.aiRatio}%`}} className="bg-rose-500 h-full" title={`AI: ${essay.originality.aiRatio}%`} />}
                                     {essay.originality.copyRatio > 0 && <div style={{width: `${essay.originality.copyRatio}%`}} className="bg-amber-500 h-full" title={`抄襲: ${essay.originality.copyRatio}%`} />}
                                     {essay.originality.originalRatio > 0 && <div style={{width: `${essay.originality.originalRatio}%`}} className="bg-emerald-500 h-full" title={`原創: ${essay.originality.originalRatio}%`} />}
                                 </div>
                                 
                                 {/* Legend & Stats */}
                                 <div className="flex flex-wrap gap-4 text-xs font-bold text-slate-600">
                                     <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> 人類原創: {essay.originality.originalRatio}%</div>
                                     <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-rose-500" /> AI 生成: {essay.originality.aiRatio}%</div>
                                     <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> 疑似抄襲: {essay.originality.copyRatio}%</div>
                                 </div>

                                 {/* Reason text if any issue */}
                                 {(essay.originality.aiRatio >= 20 || essay.originality.copyRatio >= 20) && (
                                     <div className="mt-1 p-3 bg-white rounded-xl border border-rose-100 text-xs text-slate-600 leading-relaxed shadow-sm">
                                         <span className="font-black text-rose-600 mr-2">判定原因：</span>
                                         {essay.originality.reason}
                                     </div>
                                 )}
                             </div>
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                          <div className="bg-emerald-50/60 p-5 rounded-2xl border border-emerald-200/80 shadow-inner relative group/edit">
                            <h5 className="text-xs font-black text-emerald-800 mb-3 uppercase tracking-widest flex items-center gap-2">
                              ✨ 亮點 <Pencil size={12} className="opacity-0 group-hover/edit:opacity-50 transition-opacity" />
                            </h5>
                            <textarea value={essay.highlights} onChange={(e) => handleTextChange(essay.id, 'highlights', e.target.value)} className="w-full bg-transparent border-none p-0 text-sm text-slate-800 font-medium leading-relaxed focus:ring-0 resize-none h-32 custom-scrollbar" />
                          </div>
                          <div className="bg-amber-50/60 p-5 rounded-2xl border border-amber-200/80 shadow-inner relative group/edit">
                            <h5 className="text-xs font-black text-amber-800 mb-3 uppercase tracking-widest flex items-center gap-2">
                              💡 這樣寫會更好 <Pencil size={12} className="opacity-0 group-hover/edit:opacity-50 transition-opacity" />
                            </h5>
                            <textarea value={essay.suggestions} onChange={(e) => handleTextChange(essay.id, 'suggestions', e.target.value)} className="w-full bg-transparent border-none p-0 text-sm text-slate-800 font-medium leading-relaxed focus:ring-0 resize-none h-32 custom-scrollbar" />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className={`h-32 flex items-center justify-center border-2 border-dashed rounded-2xl text-xs font-bold ${essay.status === 'error' ? 'border-rose-100 bg-rose-50 text-rose-500' : 'border-slate-100 text-slate-300'}`}>
                        {essay.status === 'processing' ? <Loader2 className="animate-spin text-indigo-500" size={24} /> : 
                         essay.status === 'error' ? 
                         <div className="flex flex-col items-center gap-2">
                             <AlertTriangle size={24} />
                             <span>{essay.response || "批改失敗"}</span>
                             <button onClick={() => processEssay(essay.id)} className="mt-2 px-4 py-2 bg-rose-200 text-rose-700 rounded-lg hover:bg-rose-300 transition-colors">重試</button>
                         </div> : "等待批改"}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </main>

      <ImageViewer src={viewingImage} onClose={() => setViewingImage(null)} />

      {batchResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-300" onClick={() => setBatchResult(null)}>
          <div className="bg-white rounded-3xl p-8 max-w-sm w-full shadow-2xl transform scale-100 animate-in zoom-in-95 duration-300" onClick={e => e.stopPropagation()}>
            <div className="text-center">
              <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-indigo-100 mb-6">
                <CheckSquare size={32} className="text-indigo-600" />
              </div>
              <h3 className="text-2xl font-black text-slate-800 mb-2">批改完成</h3>
              <p className="text-slate-500 mb-6">總共處理了 {batchResult.total} 份作文</p>
              
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-emerald-50 p-4 rounded-2xl border border-emerald-100">
                  <p className="text-3xl font-black text-emerald-600 mb-1">{batchResult.success}</p>
                  <p className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">成功</p>
                </div>
                <div className="bg-rose-50 p-4 rounded-2xl border border-rose-100">
                  <p className="text-3xl font-black text-rose-600 mb-1">{batchResult.fail}</p>
                  <p className="text-[10px] font-bold text-rose-400 uppercase tracking-widest">失敗</p>
                </div>
              </div>
              
              {/* V15.0.0: Updated Originality Dashboard in Modal to count HIGH risk overall */}
              {checkOriginality && batchResult.originalityStats && (
                 <div className="mt-4 pt-4 border-t border-slate-100">
                    <h4 className="text-sm font-bold text-slate-700 mb-3 flex items-center justify-center gap-2"><ShieldAlert size={14} className="text-slate-400"/> 高風險判定統計 ({'>'}40%)</h4>
                    <div className="flex gap-2 justify-center">
                       <div className="bg-red-50 px-3 py-2 rounded-xl border border-red-100 flex flex-col items-center w-24">
                          <span className="text-red-600 font-black text-xl">{batchResult.originalityStats.high}</span>
                          <span className="text-[10px] text-red-400 font-bold">高風險件數</span>
                       </div>
                    </div>
                 </div>
              )}

              <button onClick={() => setBatchResult(null)} className="w-full py-4 bg-slate-900 text-white rounded-2xl font-bold hover:bg-slate-800 transition-all active:scale-[0.98] mt-6">
                關閉
              </button>
            </div>
          </div>
        </div>
      )}

      {showRegradeConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-300" onClick={() => setShowRegradeConfirm(false)}>
          <div className="bg-white rounded-3xl p-8 max-w-sm w-full shadow-2xl transform scale-100 animate-in zoom-in-95 duration-300 border-2 border-amber-100" onClick={e => e.stopPropagation()}>
            <div className="text-center">
              <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-amber-100 mb-6">
                <RotateCw size={32} className="text-amber-600" />
              </div>
              <h3 className="text-2xl font-black text-slate-800 mb-2">重新批改所有作文？</h3>
              <p className="text-slate-500 mb-6 text-sm leading-relaxed">
                這將會<span className="text-amber-600 font-bold">重置所有評分結果</span>並重新開始批改。既有的評語與分數將會消失。
              </p>
              
              <div className="grid grid-cols-2 gap-3">
                <button onClick={() => setShowRegradeConfirm(false)} className="py-3 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-xl text-sm font-bold transition-colors">
                  取消
                </button>
                <button onClick={performRegradeAll} className="py-3 bg-amber-500 hover:bg-amber-600 text-white rounded-xl text-sm font-bold transition-colors shadow-lg shadow-amber-200">
                  確定重改
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showClearConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-300" onClick={() => setShowClearConfirm(false)}>
          <div className="bg-white rounded-3xl p-8 max-w-sm w-full shadow-2xl transform scale-100 animate-in zoom-in-95 duration-300 border-2 border-rose-100" onClick={e => e.stopPropagation()}>
            <div className="text-center">
              <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-rose-100 mb-6">
                <AlertTriangle size={32} className="text-rose-600" />
              </div>
              <h3 className="text-2xl font-black text-slate-800 mb-2">確定要清空嗎？</h3>
              <p className="text-slate-500 mb-6 text-sm leading-relaxed">
                此動作將會<span className="text-rose-600 font-bold">永久刪除</span>目前所有的作文資料、評分結果與分析，且<span className="underline">無法復原</span>。
              </p>
              
              <div className="grid grid-cols-2 gap-3">
                <button onClick={() => setShowClearConfirm(false)} className="py-3 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-xl text-sm font-bold transition-colors">
                  取消
                </button>
                <button onClick={performClearAll} className="py-3 bg-rose-600 hover:bg-rose-700 text-white rounded-xl text-sm font-bold transition-colors shadow-lg shadow-rose-200">
                  確定清空
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showChangelog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm px-4" onClick={() => setShowChangelog(false)}>
          <div className="bg-white rounded-3xl p-8 max-w-lg w-full max-h-[70vh] overflow-y-auto shadow-2xl" onClick={e => e.stopPropagation()}>
            <h3 className="text-xl font-black mb-6 flex items-center gap-2 text-slate-800"><History className="text-indigo-500" /> 更新日誌</h3>
            {CHANGELOG.map((log, i) => (
              <div key={i} className="mb-6 border-l-2 border-indigo-100 pl-4">
                <span className="text-sm font-black text-indigo-600 bg-indigo-50 px-2 rounded mb-2 inline-block">{log.version}</span>
                <ul className="space-y-1">{log.content.map((c, j) => <li key={j} className="text-sm text-slate-600 leading-relaxed">• {c}</li>)}</ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {toast && <Toast message={toast.msg} type={toast.type} action={toast.action} onClose={() => setToast(null)} />}
      
      <style dangerouslySetInnerHTML={{ __html: `
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@700;900&display=swap');
        .font-serif { font-family: 'Noto Serif TC', serif; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
      `}} />
    </div>
  );
};

export default App;
