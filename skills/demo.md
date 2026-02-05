"name": "sales_analytics",  # 销售分析
"description": "用于分析销售数据的数据库模式和业务逻辑，包括客户、订单和收入。",
"content": """# 销售分析模式 (Sales Analytics Schema)

## 表结构 (Tables)

### customers (客户表)
- customer_id (主键)
- name (姓名)
- email (邮箱)
- signup_date (注册日期)
- status (状态: active/inactive - 活跃/非活跃)
- customer_tier (客户等级: bronze/silver/gold/platinum - 铜/银/金/白金)

### orders (订单表)
- order_id (主键)
- customer_id (外键 -> customers)
- order_date (订单日期)
- status (状态: pending/completed/cancelled/refunded - 待处理/已完成/已取消/已退款)
- total_amount (总金额)
- sales_region (销售区域: north/south/east/west - 北/南/东/西)

### order_items (订单明细表)
- item_id (主键)
- order_id (外键 -> orders)
- product_id (产品ID)
- quantity (数量)
- unit_price (单价)
- discount_percent (折扣百分比)

## 业务逻辑 (Business Logic)

**活跃客户 (Active customers)**: status = 'active' 且 signup_date <= 当前日期 - 间隔 '90 days'

**收入计算 (Revenue calculation)**: 仅计算 status = 'completed' 的订单。使用 orders 表中的 total_amount，该金额已包含折扣。

**客户生命周期价值 (CLV)**: 一个客户所有已完成订单金额的总和。

**高价值订单 (High-value orders)**: total_amount > 1000 的订单。

## 查询示例 (Example Query)

-- 获取上一季度收入最高的前10名客户
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;
""",
