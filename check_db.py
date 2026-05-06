import sqlite3
import json
conn = sqlite3.connect('/app/memory/octts_screening.db')
cur = conn.cursor()

print("=== Top3 with selection_reason_components ===")
cur.execute('''
SELECT ts_code, name, selection_reason_components
FROM recommendation_items
WHERE run_id = (SELECT id FROM recommendation_runs ORDER BY generated_at DESC LIMIT 1)
AND recommend_rank IS NOT NULL
ORDER BY recommend_rank
''')
for row in cur.fetchall():
    print(f"\n{row[0]} | {row[1]}")
    if row[2]:
        try:
            components = json.loads(row[2])
            print(f"  fusion_70_30: {components.get('fusion_70_30')}")
            print(f"  model_score_norm: {components.get('model_score_norm')}")
            print(f"  overall_score_norm: {components.get('overall_score_norm')}")
        except:
            print(f"  raw: {row[2][:200]}")

conn.close()
