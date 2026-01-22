import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import io
import base64
import os
import platform

# 設定頁面配置
st.set_page_config(
    page_title="AI 國文作文批改助手",
    page_icon="📝",
    layout="wide"
)

# 中文字型設定函數
def setup_chinese_font():
    """設定 matplotlib 的中文字型"""
    try:
        # 嘗試使用系統內建的中文字型
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # macOS 常見中文字型
            font_candidates = [
                "PingFang TC",
                "Heiti TC",
                "STHeiti",
                "Arial Unicode MS"
            ]
        elif system == "Windows":
            font_candidates = [
                "Microsoft JhengHei",
                "Microsoft YaHei",
                "SimHei",
                "KaiTi"
            ]
        else:  # Linux
            font_candidates = [
                "WenQuanYi Micro Hei",
                "WenQuanYi Zen Hei",
                "Noto Sans CJK TC",
                "Droid Sans Fallback"
            ]
        
        # 嘗試設定字型
        for font_name in font_candidates:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                # 測試字型是否可用
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, "測試", fontsize=10)
                plt.close(fig)
                return True
            except:
                continue
        
        # 如果都失敗，使用預設設定
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False
    except Exception as e:
        st.warning(f"字型設定警告: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False

# 初始化中文字型
setup_chinese_font()

# 年級評分標準設定
GRADE_CRITERIA = {
    "國小低年級 (1-2年級)": {
        "prompt": """
請依據以下評分重點批改這篇作文：
1. 語意完整性 (40%)：句子是否完整，意思是否清楚
2. 錯字與標點 (30%)：是否有錯別字，標點符號使用是否正確
3. 觀察與情感 (30%)：是否表達出觀察與感受

請以鼓勵為主，給予建設性的建議。
""",
        "dimensions": ["語意完整性", "錯字與標點", "觀察與情感"]
    },
    "國小中年級 (3-4年級)": {
        "prompt": """
請依據以下評分重點批改這篇作文：
1. 段落結構 (30%)：段落是否清楚，是否有適當分段
2. 想像力與修辭 (30%)：是否運用想像力，是否有使用修辭技巧
3. 流暢度 (20%)：文句是否流暢自然
4. 標點錯字 (20%)：標點符號與錯別字

請給予具體的改進建議。
""",
        "dimensions": ["段落結構", "想像力與修辭", "流暢度", "標點錯字"]
    },
    "國小高年級 (5-6年級)": {
        "prompt": """
請依據以下評分重點批改這篇作文：
1. 篇章結構 (起承轉合) (30%)：文章結構是否完整，是否有起承轉合
2. 立意取材 (30%)：主題是否明確，取材是否適當
3. 修辭潤飾 (20%)：修辭技巧的運用與文辭的潤飾
4. 邏輯思考 (20%)：論述是否合乎邏輯

請給予專業的評語與建議。
""",
        "dimensions": ["篇章結構", "立意取材", "修辭潤飾", "邏輯思考"]
    },
    "國中 (7-9年級)": {
        "prompt": """
請依據以下評分重點批改這篇作文：
1. 結構完整性 (25%)：文章結構是否完整，段落安排是否合理
2. 主旨明確度 (25%)：主題是否明確，中心思想是否清楚
3. 文辭優美度 (25%)：文辭是否優美，修辭運用是否得當
4. 立意與取材 (25%)：立意是否深刻，取材是否豐富

請給予深入的評析與建議。
""",
        "dimensions": ["結構完整性", "主旨明確度", "文辭優美度", "立意與取材"]
    },
    "高中 (10-12年級)": {
        "prompt": """
請依據以下評分重點批改這篇作文：
1. 思辨能力 (30%)：思考是否深入，是否有獨到見解
2. 藝術價值 (25%)：文辭的藝術性，修辭的運用
3. 社會關懷/生命體悟 (25%)：是否展現對社會或生命的關懷與體悟
4. 論證邏輯 (20%)：論證是否嚴謹，邏輯是否清晰

請給予專業且深入的評析。
""",
        "dimensions": ["思辨能力", "藝術價值", "社會關懷/生命體悟", "論證邏輯"]
    }
}

def generate_system_prompt(grade_level):
    """根據年級生成系統提示詞"""
    criteria = GRADE_CRITERIA[grade_level]
    return f"""
你是一位資深的國文老師，專門批改學生作文。

{criteria['prompt']}

【重要要求】
1. 請先進行 OCR 辨識，將圖片中的作文文字完整提取出來。
2. 請依據上述評分標準，對這篇作文進行詳細批改。
3. 請以 JSON 格式回傳結果，格式如下：

{{
  "full_text": "辨識出的作文全文（保留段落格式）",
  "scores": {{
    "{criteria['dimensions'][0]}": 85,
    "{criteria['dimensions'][1] if len(criteria['dimensions']) > 1 else '其他'}": 90,
    "{criteria['dimensions'][2] if len(criteria['dimensions']) > 2 else '其他'}": 80,
    "{criteria['dimensions'][3] if len(criteria['dimensions']) > 3 else '其他'}": 88
  }},
  "total_score": 88,
  "comment_summary": "一句話短評（20字以內）",
  "detailed_review": "完整的 Markdown 格式評語，包含：\\n- 優點與亮點\\n- 需要改進的地方\\n- 具體建議"
}}

請確保：
- scores 中的維度名稱必須與上述評分重點完全一致
- 分數範圍為 0-100
- total_score 為所有維度分數的平均值（四捨五入）
- 只回傳 JSON，不要有其他文字說明
"""

def analyze_essay(api_key, images, grade_level):
    """使用 Gemini API 分析作文"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_prompt = generate_system_prompt(grade_level)
        
        # 構建提示詞
        prompt = f"{system_prompt}\n\n請批改以下作文圖片："
        
        # 準備內容：提示詞 + 圖片（Gemini API 可以直接接受 PIL Image 對象）
        content = [prompt] + images
        
        # 調用 API
        response = model.generate_content(content)
        
        # 解析回應
        response_text = response.text.strip()
        
        # 嘗試提取 JSON（可能回應中有其他文字）
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("無法從回應中提取 JSON 格式")
        
        json_str = response_text[json_start:json_end]
        result = json.loads(json_str)
        
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"JSON 解析錯誤：{e}\n\n回應內容：{response_text}")
        return None
    except Exception as e:
        st.error(f"API 調用錯誤：{e}")
        return None

def create_score_card(result, grade_level):
    """生成評分圖卡"""
    try:
        scores = result['scores']
        total_score = result['total_score']
        comment_summary = result['comment_summary']
        
        # 準備數據
        dimensions = list(scores.keys())
        values = list(scores.values())
        
        # 創建圖表
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # 上半部分：雷達圖
        ax1 = fig.add_subplot(gs[0], projection='polar')
        
        # 計算角度
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 閉合
        
        values_plot = values + values[:1]  # 閉合
        
        # 繪製雷達圖
        ax1.plot(angles, values_plot, 'o-', linewidth=2, color='#4A90E2', label='評分')
        ax1.fill(angles, values_plot, alpha=0.25, color='#4A90E2')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(dimensions, fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.set_yticks([20, 40, 60, 80, 100])
        ax1.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_title(f'作文評分雷達圖 - 總分：{total_score} 分', fontsize=16, fontweight='bold', pad=20)
        
        # 下半部分：文字資訊
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        # 顯示各項分數
        score_text = "各項評分：\n"
        for dim, val in zip(dimensions, values):
            score_text += f"  • {dim}：{val} 分\n"
        
        info_text = f"{score_text}\n簡短評語：\n{comment_summary}"
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'{grade_level} 作文評分圖卡', fontsize=18, fontweight='bold', y=0.98)
        
        # 轉換為圖片
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close()
        
        return buf
        
    except Exception as e:
        st.error(f"生成評分圖卡時發生錯誤：{e}")
        return None

# 主程式
def main():
    # 側邊欄：API Key 輸入
    with st.sidebar:
        st.header("⚙️ 設定")
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="請輸入您的 Google Gemini API Key"
        )
        
        if api_key:
            st.success("✓ API Key 已設定")
        else:
            st.warning("⚠️ 請先輸入 API Key")
    
    # 主畫面
    st.title("📝 AI 國文作文批改助手 (含評分圖卡)")
    st.markdown("---")
    
    # 年級選擇
    grade_level = st.selectbox(
        "選擇年級",
        options=list(GRADE_CRITERIA.keys()),
        help="請選擇學生的年級，系統會依據年級調整評分標準"
    )
    
    st.markdown("---")
    
    # 圖片上傳
    st.subheader("📷 上傳作文圖片")
    uploaded_files = st.file_uploader(
        "請上傳作文圖片（支援 JPG/PNG，可上傳多張）",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="可以上傳多張圖片，系統會自動合併辨識"
    )
    
    # 顯示上傳的圖片預覽
    if uploaded_files:
        st.info(f"已上傳 {len(uploaded_files)} 張圖片")
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 3]:
                img = Image.open(uploaded_file)
                st.image(img, caption=f"圖片 {idx + 1}", use_container_width=True)
    
    st.markdown("---")
    
    # 開始批改按鈕
    if st.button("🚀 開始批改", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ 請先在側邊欄輸入 Google API Key")
        elif not uploaded_files:
            st.error("❌ 請先上傳作文圖片")
        else:
            with st.spinner("正在分析作文，請稍候..."):
                # 讀取圖片
                images = []
                for uploaded_file in uploaded_files:
                    img = Image.open(uploaded_file)
                    # 轉換為 RGB（如果是 RGBA）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(img)
                
                # 分析作文
                result = analyze_essay(api_key, images, grade_level)
                
                if result:
                    # 儲存結果到 session state
                    st.session_state['result'] = result
                    st.session_state['grade_level'] = grade_level
                    st.success("✓ 批改完成！")
    
    # 顯示結果
    if 'result' in st.session_state:
        result = st.session_state['result']
        grade_level = st.session_state['grade_level']
        
        st.markdown("---")
        st.subheader("📊 批改結果")
        
        # 顯示原文
        with st.expander("📄 OCR 辨識出的原文", expanded=True):
            st.text_area("", value=result.get('full_text', ''), height=200, disabled=True)
        
        # 顯示詳細評語
        st.subheader("💬 AI 評語與建議")
        st.markdown(result.get('detailed_review', ''))
        
        # 生成並顯示評分圖卡
        st.markdown("---")
        st.subheader("📈 評分圖卡")
        
        score_card = create_score_card(result, grade_level)
        
        if score_card:
            # 顯示圖卡
            score_card.seek(0)
            st.image(score_card, use_container_width=True)
            
            # 下載按鈕
            score_card.seek(0)
            st.download_button(
                label="⬇️ 下載評分圖卡 (PNG)",
                data=score_card,
                file_name=f"作文評分圖卡_{grade_level.replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
