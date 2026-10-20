"""Monochrome chat tokens, following the full-screen concept design."""

CSS = """
<style>
:root { --ink:#20232b; --muted:#777c88; --line:#e5e5e8; --accent:#272a33; --sidebar:#f7f7f8; }
html, body, [data-testid="stApp"] { font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif; color:var(--ink); background:#fff; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#fff; }
[data-testid="stMainBlockContainer"] { max-width:1120px; padding:1.4rem 2.5rem 3.5rem; }
[data-testid="stHeader"] { height:0; background:transparent !important; }
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], #MainMenu { display:none; }
[data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--line); }
[data-testid="stSidebarHeader"] { height:1.5rem; min-height:1.5rem; }
[data-testid="stSidebarUserContent"] { padding:0 1rem 7.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.45rem; }
[data-testid="stVerticalBlock"] { gap:.75rem; }
.app-brand { display:flex; align-items:center; gap:12px; font-size:21px; font-weight:650; letter-spacing:-.4px; margin:0 8px 22px; }
.app-brand svg { width:27px; height:30px; flex-shrink:0; }
.history-label { color:var(--muted); font-size:12px; margin:25px 10px 8px; }
[data-testid="stSidebar"] button { justify-content:flex-start; min-height:42px; padding:9px 12px; border-radius:10px; }
[data-testid="stSidebar"] button p { font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; text-align:left; }
.st-key-new_chat button { justify-content:center; border-color:#c5c6cc; background:transparent; }
.st-key-conversation_list button { border:0; }
.st-key-conversation_list button[kind="primary"] { background:#e9e9ed; color:var(--ink); }
.st-key-conversation_list button[kind="primary"]:hover { background:#e2e2e7; color:var(--ink); }
.st-key-conversation_list button[kind="tertiary"]:hover { background:#ededf0; color:var(--ink); }
.st-key-sidebar_footer { position:fixed; bottom:0; left:0; width:249px; padding:12px 16px 18px; background:var(--sidebar); z-index:3; }
.local-note { color:var(--muted); font-size:11px; margin:0 0 0 39px; }
.document-title { font-size:20px !important; font-weight:650 !important; padding:0 !important; margin:0 !important; line-height:1.5 !important; overflow-wrap:anywhere; }
.document-subtitle { font-size:12px; color:var(--muted); margin:4px 0 0; }
.header-rule { height:1px; background:var(--line); margin:0 -2.5rem 10px; }
.original-text { color:var(--ink); font-size:15px; line-height:1.9; white-space:pre-wrap; overflow-wrap:anywhere; tab-size:4; }
[data-testid="stPopover"] button { border:none; background:transparent; border-radius:8px; padding:6px; }
[data-testid="stPopoverButton"] > div:last-child { display:none; }
.st-key-chat_history { max-width:790px; margin:0 auto; padding-top:4px; }
[data-testid="stChatMessage"] { background:transparent; padding:14px 0; gap:16px; }
[data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarCustom"] { background:#f3f4f6; color:var(--ink); width:38px; height:38px; border-radius:50%; border:0; }
[data-testid="stChatMessageContent"] { min-width:0; }
[data-testid="stChatMessageContent"] p { font-size:15px; line-height:1.85; }
[data-testid="stChatMessageContent"] [data-testid="stCaptionContainer"] p { font-size:12px; color:var(--muted); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { justify-content:flex-end; padding:20px 0 8px; }
[data-testid="stChatMessageAvatarUser"] { display:none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { flex:0 1 auto; width:auto; max-width:85%; margin:0; background:#f0f0f3; border-radius:18px; padding:12px 18px; }
.user-question, .answer-text { font-size:16px; line-height:1.85; overflow-wrap:anywhere; white-space:pre-wrap; }
.answer-text { line-height:1.6; padding-top:2px; padding-bottom:2px; white-space:normal; }
.answer-text p { margin:0; font-size:inherit; line-height:1.6; white-space:pre-wrap; }
.answer-text p + p { margin-top:.5em; }
[data-testid="stChatMessageContent"] [data-testid="stExpander"] { border:0; border-radius:8px; background:transparent; }
[data-testid="stChatMessageContent"] [data-testid="stExpander"] > details { border:0; border-radius:8px; }
[data-testid="stChatMessageContent"] [data-testid="stExpander"] summary { padding:4px 0; color:var(--muted); background:transparent; min-height:32px; }
[data-testid="stChatMessageContent"] [data-testid="stExpander"] summary p { font-size:12px; }
[data-testid="stChatMessageContent"] [data-testid="stExpanderDetails"] { border-left:2px solid #e8e8ec; margin:8px 0; padding:8px 0 8px 16px; }
[data-testid="stBottom"] { background:#fff; }
[data-testid="stBottomBlockContainer"] { max-width:870px; padding:16px 40px 8px; background:#fff; }
[data-testid="stBottomBlockContainer"]:after { content:"回答依据当前文档，每个问题独立处理"; display:block; color:var(--muted); font-size:11px; padding:9px 2px 0; }
[data-testid="stChatInput"] { background:#fff; border:1px solid #c9cbd2; border-radius:22px; box-shadow:0 2px 8px #20232b08; padding:12px 16px; min-height:112px; }
[data-testid="stChatInput"]:focus-within { border-color:#8e929d; box-shadow:0 0 0 1px #8e929d; }
[data-testid="stChatInput"] > div { position:relative; background:transparent; border:0; padding:0; min-height:86px; border-radius:0; }
[data-testid="stChatInput"] [data-baseweb="textarea"], [data-testid="stChatInput"] [data-baseweb="base-input"] { background:transparent; border:0; }
[data-testid="stChatInput"] textarea { font-family:inherit; font-size:15px; line-height:1.6; min-height:86px; padding:0 4px 38px !important; }
[data-testid="stChatInput"] > div > div:first-child { position:absolute; left:0; bottom:0; height:32px; padding:0; z-index:1; }
[data-testid="stChatInput"] > div > div:last-child { position:absolute; right:0; bottom:0; }
[data-testid="stChatInputSubmitButton"] { background:#272a33; color:#fff; border-radius:50%; width:32px; height:32px; }
[data-testid="stChatInputSubmitButton"]:disabled { background:#e5e5e8; color:#a4a5af; }
[data-testid="stChatInputSubmitButton"] svg { display:none; }
[data-testid="stChatInputSubmitButton"]:after { content:""; width:20px; height:20px; background:currentColor; mask:url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"%3E%3Cpath d="M12 19V5M5 12l7-7 7 7" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/%3E%3C/svg%3E') center/contain no-repeat; }
[data-testid="stChatInput"] button { border-radius:50%; }
.st-key-empty_state { max-width:560px; margin:clamp(36px,12vh,130px) auto 40px; text-align:center; }
.st-key-empty_state h1 { font-size:28px !important; font-weight:600; line-height:1.5; letter-spacing:-.8px; padding:0; margin:16px 0 8px; }
.st-key-empty_state p { color:var(--muted); font-size:14px; }
.empty-document { width:58px; height:58px; border-radius:16px; display:flex; align-items:center; justify-content:center; margin:auto; background:#f4f4f6; }
.empty-document svg { width:29px; height:29px; }
.st-key-empty_state [data-testid="stFileUploader"] { text-align:left; margin-top:22px; }
[data-testid="stFileUploaderDropzone"] { border:1px dashed #d4d5db; background:#fafafa; border-radius:14px; padding:24px 18px; }
button, input, textarea, select { font-family:inherit !important; }
button { border-radius:9px; }
button p { font-size:14px; }
[data-testid="stForm"] { border:0; padding:0; }
h1 { font-size:28px !important; letter-spacing:-.5px; }
h2 { font-size:20px !important; }
h3 { font-size:17px !important; }
.evidence-text { color:var(--ink); line-height:1.9; font-size:14px; margin:8px 0 12px; overflow-wrap:anywhere; white-space:pre-wrap; }
.source-label { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
.evidence-heading { display:flex; justify-content:space-between; gap:12px; }
.evidence-heading strong { font-size:18px; }
.score { font-size:13px; color:var(--muted); }
mark { background:#e7ecef; color:#20232b; padding:1px 0; border-radius:2px; }
[data-testid="stExpander"] { border-color:var(--line); border-radius:9px; }
[data-testid="stTabs"] button p { font-size:14px; }
.stats { display:flex; gap:24px; flex-wrap:wrap; }
.stat small { display:block; color:var(--muted); }
@media(min-width: 769px) {
 [data-testid="stSidebar"] { width:250px !important; min-width:250px !important; max-width:250px !important; }
}
@media(max-width: 768px) {
 [data-testid="stMainBlockContainer"] { padding:3rem 1rem 2rem; }
 [data-testid="stSidebar"] { max-width:280px; }
 .st-key-sidebar_footer { width:249px; }
 .header-rule { margin-left:-1rem; margin-right:-1rem; }
 .document-title { font-size:17px !important; }
 .document-subtitle { font-size:11px; }
 [data-testid="stHorizontalBlock"]:has(.document-title) { flex-wrap:nowrap; gap:.5rem; }
 [data-testid="stHorizontalBlock"]:has(.document-title) > [data-testid="stColumn"]:first-child { min-width:0; flex:1 1 0; }
 [data-testid="stHorizontalBlock"]:has(.document-title) > [data-testid="stColumn"]:last-child { min-width:32px; flex:0 0 32px; }
 [data-testid="stBottomBlockContainer"] { padding:10px 16px 12px; }
 [data-testid="stBottomBlockContainer"]:after { font-size:10px; }
 [data-testid="stChatInput"] { min-height:92px; border-radius:20px; }
 [data-testid="stChatMessage"] { gap:10px; }
 [data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarCustom"] { width:30px; height:30px; }
 .user-question, .answer-text { font-size:15px; }
 .st-key-empty_state { margin-top:12vh; }
 .st-key-empty_state h1 { font-size:24px !important; }
 [data-testid="stFileUploaderDropzone"] { padding:16px 12px; }
}
@media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; transition:none !important; } }
</style>
"""
