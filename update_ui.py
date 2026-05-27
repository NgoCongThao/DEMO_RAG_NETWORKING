import re
import codecs

with codecs.open('app.py', 'r', 'utf-8') as f:
    content = f.read()

# 1. Replace CSS
new_css = '''<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar Logo */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0 20px 0;
    margin-bottom: 20px;
}
.sidebar-logo i { font-size: 24px; color: #10a37f; }
.sidebar-logo .logo-text { font-size: 18px; font-weight: 600; }

/* Hide Streamlit elements */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
</style>'''

content = re.sub(r'<style>.*?</style>', new_css, content, flags=re.DOTALL)

# 2. Replace Sidebar Logo
content = re.sub(
    r'<div class="sidebar-logo">.*?</div>\s*</div>',
    '<div class="sidebar-logo">\\n        <i class="fa-solid fa-network-wired"></i>\\n        <div class="logo-text">RAG Networking</div>\\n    </div>',
    content,
    flags=re.DOTALL
)

# 3. Remove Thông số kỹ thuật
# It starts with st.markdown("**⚙️ Thông số kỹ thuật**") and ends with st.divider()
content = re.sub(r'        st\.markdown\("\*\*⚙️ Thông số kỹ thuật\*\*"\).*?st\.divider\(\)\n', '', content, flags=re.DOTALL)

# 4. Replace avatars and emojis
content = content.replace('avatar="🧑‍💻"', 'avatar="user"')
content = content.replace('avatar="🌐"', 'avatar="assistant"')
content = content.replace('"🧑‍💻" if message["role"] == "user" else "🌐"', '"user" if message["role"] == "user" else "assistant"')

# Header rewrite
old_header = '''<div class="page-header">
    <span style="font-size:36px;">🤖</span>
    <div>
        <h1>Trợ lý ảo Mạng Máy Tính</h1>
        <p class="subtitle">RAG · Gemini 2.5 Flash · Hybrid Search (Vector + BM25)</p>
    </div>
</div>'''
new_header = '''<div class="page-header">
    <i class="fa-solid fa-robot" style="font-size:36px; color:#10a37f; margin-right: 15px;"></i>
    <div style="display: inline-block; vertical-align: middle;">
        <h1 style="margin: 0;">Trợ lý ảo Mạng Máy Tính</h1>
        <p class="subtitle" style="margin: 0; color: gray;">RAG · Gemini 2.5 Flash</p>
    </div>
</div>'''
content = content.replace(old_header, new_header)

with codecs.open('app.py', 'w', 'utf-8') as f:
    f.write(content)

print('Success')
