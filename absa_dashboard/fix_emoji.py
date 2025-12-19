# Fix broken emoji in dashboard.py
import re

with open('dashboard.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace the broken navigation list
old_nav = '''["📊 Sentiment Overview", "🔍 Product Explorer", "🏷️ Aspect Analysis", 
     "📈 Product Deep Dive", "� Sentiment Forecasting", "�🚨 Alerts & Anomalies"]'''

new_nav = '''["📊 Sentiment Overview", "🔍 Product Explorer", "🏷️ Aspect Analysis", 
     "📈 Product Deep Dive", "🔮 Sentiment Forecasting", "🚨 Alerts & Anomalies"]'''

if old_nav in content:
    content = content.replace(old_nav, new_nav)
    print("✅ Found and replaced broken emojis")
else:
    # Try line by line replacement
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Sentiment Forecasting' in line and '�' in line:
            lines[i] = line.replace('�', '🔮')
            print(f"✅ Fixed line {i+1}: Sentiment Forecasting emoji")
        if 'Alerts & Anomalies' in line and '�' in line:
            lines[i] = line.replace('�🚨', '🚨')
            print(f"✅ Fixed line {i+1}: Alerts emoji")
    content = '\n'.join(lines)

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Dashboard emojis fixed!")
