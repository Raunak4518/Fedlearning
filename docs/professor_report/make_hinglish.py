"""Assemble build_hinglish.js from the helpers of build_lean.js and hinglish_content.js, then: node build_hinglish.js"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(HERE, "build_lean.js"), encoding="utf-8").read()
c = open(os.path.join(HERE, "hinglish_content.js"), encoding="utf-8").read()
c = c.replace('["#", "Component", "Ek line mein", "Page"]', '["#", "Component", "Ek line mein"]')
c = re.sub(r'(\["\d", "[^"]*", "[^"]*"), "\d+"\],', r'\1],', c)
c = c.replace('], [0.4, 2.6, 5.4, 0.8]));', '], [0.4, 2.8, 6.4]));')
head = s[:s.index("// ---------- content")]
tail = s[s.index("// ---------- document"):]
tail = tail.replace('title: "Ours-A for Generator-Based Heterogeneous FL"', 'title: "Ours-A notes (Hinglish)"')
tail = tail.replace('"Ours-A · Research Progress"', '"Ours-A · personal notes"')
tail = tail.replace('"OursA_Backup_Handout.docx"', '"OursA_Hinglish_Notes.docx"')
open(os.path.join(HERE, "build_hinglish.js"), "w", encoding="utf-8").write(head + "// ---------- content\nconst C = [];\n" + c + "\n" + tail)
print("wrote build_hinglish.js")
