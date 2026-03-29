⚠️ 当前工具不可用，我先给你一条可参考的 Holter SQL。

```
SELECT
  id,
  user_id,
  use_day,
  begin_date_time,
  end_date_time,
  is_uploaded,
  report_status,
  holter_type,
  add_time,
  update_time
FROM t_holter_use_record
WHERE use_day BETWEEN '2026-03-29' AND '2026-03-29'
ORDER BY id DESC
LIMIT 5;
```

说明：如果该表名在你的库不存在，请替换为实际 Holter 业务表。