# ABSA Dashboard: Visual Data Flow Guide
# ========================================

"""
This file shows EXACTLY what happens to your data at each step.
Follow along with real examples!
"""

# ============================================================
# STEP 1: RAW DATA (What you start with)
# ============================================================

print("\n" + "="*60)
print("STEP 1: RAW DATA")
print("="*60)

raw_absa = """
Your ABSA results (df):
┌────────────┬─────────────┬──────────────┬───────────┬────────────┐
│ review_id  │ parent_asin │ aspect_term  │ sentiment │ confidence │
├────────────┼─────────────┼──────────────┼───────────┼────────────┤
│ 123        │ B08X1Y2Z3   │ color        │ Positive  │ 0.9        │
│ 124        │ B08X1Y2Z3   │ colors       │ Positive  │ 0.7        │
│ 125        │ B09A2B3C4   │ smell        │ Negative  │ 0.8        │
│ 126        │ B09A2B3C4   │ scent        │ Negative  │ 0.6        │
│ 127        │ B09A2B3C4   │ price        │ Positive  │ 0.5        │
└────────────┴─────────────┴──────────────┴───────────┴────────────┘
"""

raw_products = """
Your product metadata (product_df):
┌─────────────┬──────────────────┬───────────────┬────────────────┐
│ parent_asin │ title            │ main_category │ average_rating │
├─────────────┼──────────────────┼───────────────┼────────────────┤
│ B08X1Y2Z3   │ Blue Nail Polish │ Beauty        │ 4.5            │
│ B09A2B3C4   │ Rose Perfume     │ Fragrance     │ 3.8            │
└─────────────┴──────────────────┴───────────────┴────────────────┘
"""

print(raw_absa)
print(raw_products)

print("\n⚠️  PROBLEMS:")
print("  1. 'color' and 'colors' are separate (should be same)")
print("  2. 'smell' and 'scent' are separate (should be same)")
print("  3. No product titles in ABSA data (just ASINs)")


# ============================================================
# STEP 2: ASPECT NORMALIZATION
# ============================================================

print("\n" + "="*60)
print("STEP 2: ASPECT NORMALIZATION")
print("="*60)

normalized = """
After normalize_aspect_terms():
┌────────────┬─────────────┬──────────────┬─────────────────────┬───────────┬────────────┐
│ review_id  │ parent_asin │ aspect_term  │ aspect_term_        │ sentiment │ confidence │
│            │             │              │ normalized          │           │            │
├────────────┼─────────────┼──────────────┼─────────────────────┼───────────┼────────────┤
│ 123        │ B08X1Y2Z3   │ color        │ color               │ Positive  │ 0.9        │
│ 124        │ B08X1Y2Z3   │ colors       │ color ◄── MAPPED!   │ Positive  │ 0.7        │
│ 125        │ B09A2B3C4   │ smell        │ smell               │ Negative  │ 0.8        │
│ 126        │ B09A2B3C4   │ scent        │ smell ◄── MAPPED!   │ Negative  │ 0.6        │
│ 127        │ B09A2B3C4   │ price        │ price               │ Positive  │ 0.5        │
└────────────┴─────────────┴──────────────┴─────────────────────┴───────────┴────────────┘
"""

print(normalized)

print("\n✅ FIXED:")
print("  - 'color' and 'colors' → both mapped to 'color'")
print("  - 'smell' and 'scent' → both mapped to 'smell'")
print("  - Now we can count: color=2, smell=2, price=1")


# ============================================================
# STEP 3: MERGE WITH PRODUCT DATA
# ============================================================

print("\n" + "="*60)
print("STEP 3: MERGE WITH PRODUCT DATA")
print("="*60)

merged = """
After merge_dataframes():
┌────────────┬─────────────┬──────────────┬───────────┬──────────────────┬───────────────┐
│ review_id  │ parent_asin │ aspect_term_ │ sentiment │ title            │ main_category │
│            │             │ normalized   │           │                  │               │
├────────────┼─────────────┼──────────────┼───────────┼──────────────────┼───────────────┤
│ 123        │ B08X1Y2Z3   │ color        │ Positive  │ Blue Nail Polish │ Beauty        │ ◄── ADDED!
│ 124        │ B08X1Y2Z3   │ color        │ Positive  │ Blue Nail Polish │ Beauty        │ ◄── ADDED!
│ 125        │ B09A2B3C4   │ smell        │ Negative  │ Rose Perfume     │ Fragrance     │ ◄── ADDED!
│ 126        │ B09A2B3C4   │ smell        │ Negative  │ Rose Perfume     │ Fragrance     │ ◄── ADDED!
│ 127        │ B09A2B3C4   │ price        │ Positive  │ Rose Perfume     │ Fragrance     │ ◄── ADDED!
└────────────┴─────────────┴──────────────┴───────────┴──────────────────┴───────────────┘
"""

print(merged)

print("\n✅ FIXED:")
print("  - Now each review has product title!")
print("  - Now each review has category!")
print("  - Can filter/group by category")
print("  - Can show product names in dashboard")


# ============================================================
# STEP 4: DATA CLEANING & DERIVED FIELDS
# ============================================================

print("\n" + "="*60)
print("STEP 4: DATA CLEANING & DERIVED FIELDS")
print("="*60)

cleaned = """
After clean_merged_data():
┌────────────┬─────────────┬─────────────┬───────────┬────────────┬─────────────────┬──────────┐
│ review_id  │ parent_asin │ aspect_term │ sentiment │ confidence │ sentiment_score │ date     │
│            │             │ _normalized │           │            │                 │          │
├────────────┼─────────────┼─────────────┼───────────┼────────────┼─────────────────┼──────────┤
│ 123        │ B08X1Y2Z3   │ color       │ Positive  │ 0.9        │ +0.9 ◄── NEW!   │ 2020-01  │ ◄── NEW!
│ 124        │ B08X1Y2Z3   │ color       │ Positive  │ 0.7        │ +0.7 ◄── NEW!   │ 2020-02  │ ◄── NEW!
│ 125        │ B09A2B3C4   │ smell       │ Negative  │ 0.8        │ -0.8 ◄── NEW!   │ 2020-03  │ ◄── NEW!
│ 126        │ B09A2B3C4   │ smell       │ Negative  │ 0.6        │ -0.6 ◄── NEW!   │ 2020-04  │ ◄-- NEW!
│ 127        │ B09A2B3C4   │ price       │ Positive  │ 0.5        │ +0.5 ◄-- NEW!   │ 2020-05  │ ◄-- NEW!
└────────────┴─────────────┴─────────────┴───────────┴────────────┴─────────────────┴──────────┘
"""

print(cleaned)

print("\n✅ ADDED:")
print("  - sentiment_score: Numeric version (-1 to +1)")
print("  - date: Extracted from timestamp")
print("  - Also added: year_month, week, is_positive, is_negative")
print("  - Removed: Duplicates, low-confidence (<0.5), generic aspects")


# ============================================================
# STEP 5: AGGREGATION EXAMPLES
# ============================================================

print("\n" + "="*60)
print("STEP 5: WHAT YOU CAN DO NOW (Examples)")
print("="*60)

print("\n📊 Example 1: Count aspects per product")
print("-" * 40)
example1 = """
df.groupby(['parent_asin', 'aspect_term_normalized']).size()

Result:
parent_asin  aspect_term_normalized
B08X1Y2Z3    color                     2  ← "color" mentioned 2x
B09A2B3C4    smell                     2  ← "smell" mentioned 2x
             price                     1  ← "price" mentioned 1x
"""
print(example1)

print("\n📈 Example 2: Average sentiment per product")
print("-" * 40)
example2 = """
df.groupby('parent_asin')['sentiment_score'].mean()

Result:
parent_asin
B08X1Y2Z3    0.80  ← Very positive! (0.9 + 0.7) / 2
B09A2B3C4   -0.30  ← Negative (-0.8 - 0.6 + 0.5) / 3
"""
print(example2)

print("\n📅 Example 3: Sentiment trend over time")
print("-" * 40)
example3 = """
df.groupby('date')['sentiment_score'].mean()

Result:
date
2020-01    0.90  ← Good start
2020-02    0.70  ← Still positive
2020-03   -0.80  ← Uh oh, negative!
2020-04   -0.60  ← Still negative
2020-05    0.50  ← Recovering
"""
print(example3)

print("\n🎯 Example 4: Top products by review count")
print("-" * 40)
example4 = """
df['parent_asin'].value_counts()

Result:
B09A2B3C4    3  ← Most reviewed
B08X1Y2Z3    2  ← Second
"""
print(example4)


# ============================================================
# STEP 6: DASHBOARD VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("STEP 6: DASHBOARD VISUALIZATION")
print("="*60)

viz_example = """
Now in dashboard.py, you can create charts like:

1. KPI Cards:
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ Total       │  │ Unique      │  │ Avg         │
   │ Reviews     │  │ Products    │  │ Confidence  │
   │   5         │  │   2         │  │   0.70      │
   └─────────────┘  └─────────────┘  └─────────────┘

2. Pie Chart (Sentiment Distribution):
         Positive
           40%
        ╱      ╲
   Neutral    Negative
     20%        40%

3. Line Chart (Sentiment Over Time):
   1.0 │           ╱╲
   0.5 │      ╱╲  ╱  ╲
   0.0 │─────────────────────
  -0.5 │          ╲    ╱
  -1.0 │           ╲  ╱
       └─────────────────────
       Jan Feb Mar Apr May

4. Heatmap (Aspect × Product):
                 Product A  Product B
   color          +0.8       +0.5     🟢 Green
   smell          -0.7       -0.6     🔴 Red
   price          +0.5       +0.3     🟡 Yellow
"""

print(viz_example)


# ============================================================
# STEP 7: ALERT DETECTION
# ============================================================

print("\n" + "="*60)
print("STEP 7: ALERT DETECTION")
print("="*60)

alert_example = """
alert_system.py monitors for problems:

1. Sentiment Drop Example:
   
   Product B09A2B3C4 (Rose Perfume):
   
   Week 1: +0.6  🟢 Good
   Week 2: +0.4  🟡 OK
   Week 3: +0.1  🟡 Declining
   Week 4: -0.3  🔴 ALERT! Dropped 67%!
   
   ⚠️  ALERT TRIGGERED:
   - Product: Rose Perfume (B09A2B3C4)
   - Change: -67%
   - Top negative aspects: smell, price, packaging

2. Emerging Aspect Example:
   
   Last 30 days:
   - New aspect detected: "leak" (50 mentions)
   - New aspect detected: "broken" (30 mentions)
   
   ⚠️  ALERT: New quality issues appearing!

3. Rating Divergence Example:
   
   Product: Blue Nail Polish
   - Amazon Rating: 4.5 ⭐⭐⭐⭐½
   - Sentiment Rating: 2.5 ⭐⭐½ (from reviews)
   - Divergence: 2.0 stars!
   
   ⚠️  ALERT: Recent reviews much worse than old reviews!
"""

print(alert_example)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY: THE COMPLETE DATA JOURNEY")
print("="*60)

summary = """
1. RAW DATA (messy)
   ↓
2. NORMALIZATION (clean aspects)
   ↓
3. MERGING (add product info)
   ↓
4. CLEANING (remove noise, add derived fields)
   ↓
5. PREPROCESSING COMPLETE ✓
   ↓
6. DASHBOARD (visualize)
   ↓
7. ALERTS (monitor)
   ↓
8. TOPICS (discover themes)
   ↓
9. INSIGHTS! 🎯

From 68,772 messy rows → Clean, analyzable, visualized data!
"""

print(summary)

print("\n" + "="*60)
print("Now check out CODE_GUIDE.md for detailed explanations!")
print("="*60)
