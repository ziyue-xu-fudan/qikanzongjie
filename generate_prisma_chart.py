import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
from matplotlib import rcParams

# Configuration for Fonts
rcParams['font.sans-serif'] = ['Arial'] # Default fallback

def get_font(lang):
    if lang == 'cn':
        return 'Heiti TC' # macOS standard Chinese font
    return 'Arial'

def draw_box(ax, x, y, w, h, text, fontname, color='#EBF5FB', edge='#2E86C1'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", 
                                  linewidth=2, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, 
            fontweight='bold', color='#1B4F72', fontname=fontname)
    return x + w/2, y

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#5D6D7E"))

def generate_chart(lang='en', total=619, excluded_rct=85, included_final=0):
    # Text Dictionaries
    texts = {
        'en': {
            'title': "PRISMA Flow Diagram: Non-RCT Oncology Research",
            'box1': f"Records identified from PubMed\n(NEJM, Lancet, JAMA, BMJ)\n(n = {total})",
            'box2': f"Records excluded:\nRCTs / Phase 3 Trials\n(n = {excluded_rct})",
            'box3': f"Studies included in\nfinal analysis\n(n = {included_final})"
        },
        'cn': {
            'title': "PRISMA 文献筛选流程图：非随机对照肿瘤研究",
            'box1': f"PubMed 数据库检出文献\n(NEJM, Lancet, JAMA, BMJ)\n(n = {total})",
            'box2': f"排除文献：\nRCT / 三期临床试验\n(n = {excluded_rct})",
            'box3': f"最终纳入分析\n(n = {included_final})"
        }
    }
    
    t = texts[lang]
    font = get_font(lang)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, t['title'], ha='center', fontsize=16, fontweight='bold', 
            color='#2C3E50', fontname=font)

    # --- Vertical 3-Box Layout ---
    
    # Box 1: Top (Identified)
    draw_box(ax, 2.5, 7.5, 5, 1.2, t['box1'], font, color='#EBF5FB', edge='#2E86C1')
    
    # Arrow 1
    draw_arrow(ax, 5, 7.5, 5, 6.2)
    
    # Box 2: Middle (Excluded) - Using a different color to indicate "Action/Filter"
    draw_box(ax, 2.5, 5, 5, 1.2, t['box2'], font, color='#FDEDEC', edge='#C0392B')
    
    # Arrow 2
    draw_arrow(ax, 5, 5, 5, 3.7)
    
    # Box 3: Bottom (Final)
    draw_box(ax, 2.5, 2.5, 5, 1.2, t['box3'], font, color='#D4EFDF', edge='#27AE60')

    plt.tight_layout()
    output_filename = f"PRISMA_{lang.upper()}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {output_filename}")
    plt.close()

def main():
    # Real Data
    total_identified = 547
    removed_count = 344
    screened_count = 203 # This is the final number
    
    generate_chart('en', total=total_identified, excluded_rct=removed_count, included_final=screened_count)
    generate_chart('cn', total=total_identified, excluded_rct=removed_count, included_final=screened_count)

if __name__ == "__main__":
    main()
